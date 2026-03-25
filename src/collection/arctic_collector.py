"""
Colector histórico usando Arctic Shift API.

Arctic Shift es un archivo público de Reddit que permite descargar posts
y comentarios por rango de fechas exacto — sin las limitaciones de 1000
posts de la API oficial de Reddit.

API base: https://arctic-shift.photon-reddit.com/api
Endpoints usados:
  GET /posts/search?subreddit=X&after=EPOCH&before=EPOCH&limit=100
  GET /comments/search?link_id=POST_ID&limit=100

Paginación: se repite la query con after=último_created_utc hasta
agotar los resultados del día.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from loguru import logger

from src.database.db_manager import DatabaseManager

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
MAX_PER_PAGE = 100        # Límite máximo por request de Arctic Shift
REQUEST_SLEEP = 0.5       # Segundos entre requests (respetar rate limit)
COMMENTS_PER_POST = 200   # Comentarios máximos por post


class ArcticCollector:
    """
    Recolector histórico de datos de Reddit usando Arctic Shift API.

    Permite obtener posts y comentarios distribuidos uniformemente
    a lo largo de N días, necesario para el análisis temporal de Δ.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "reddit-political-analysis/1.0 (academic research)"
        })

    def _get_posts_for_day(
        self,
        subreddit: str,
        day_start_ts: int,
        day_end_ts: int,
    ) -> list[dict]:
        """
        Descarga TODOS los posts de un subreddit en un rango de tiempo,
        paginando hasta agotar los resultados.
        """
        url = f"{ARCTIC_BASE}/posts/search"
        all_posts = []
        after_ts = day_start_ts

        while True:
            params = {
                "subreddit": subreddit,
                "after": after_ts,
                "before": day_end_ts,
                "limit": MAX_PER_PAGE,
            }
            try:
                r = self.session.get(url, params=params, timeout=20)
                r.raise_for_status()
                posts = r.json().get("data", [])
            except Exception as e:
                logger.warning(f"  Error en Arctic Shift /posts/search: {e}")
                break

            if not posts:
                break

            all_posts.extend(posts)

            # Si devolvió menos del máximo, no hay más páginas
            if len(posts) < MAX_PER_PAGE:
                break

            # Paginar: avanzar el after al último post devuelto
            last_ts = int(float(posts[-1]["created_utc"]))
            if last_ts <= after_ts:
                break
            after_ts = last_ts
            time.sleep(REQUEST_SLEEP)

        return all_posts

    def _get_comments_for_post(self, post_id: str) -> list[dict]:
        """
        Descarga hasta COMMENTS_PER_POST comentarios de un post.
        """
        url = f"{ARCTIC_BASE}/comments/search"
        all_comments = []
        after_ts = 0

        while len(all_comments) < COMMENTS_PER_POST:
            params = {
                "link_id": post_id,
                "limit": min(MAX_PER_PAGE, COMMENTS_PER_POST - len(all_comments)),
            }
            if after_ts:
                params["after"] = after_ts

            try:
                r = self.session.get(url, params=params, timeout=20)
                r.raise_for_status()
                comments = r.json().get("data", [])
            except Exception as e:
                logger.warning(f"  Error obteniendo comentarios del post {post_id}: {e}")
                break

            if not comments:
                break

            all_comments.extend(comments)

            if len(comments) < MAX_PER_PAGE:
                break

            last_ts = int(float(comments[-1]["created_utc"]))
            if last_ts <= after_ts:
                break
            after_ts = last_ts
            time.sleep(REQUEST_SLEEP)

        return all_comments

    def _extract_post_data(self, p: dict, subreddit: str) -> dict:
        return {
            "id": p["id"],
            "subreddit": subreddit,
            "title": p.get("title", ""),
            "selftext": p.get("selftext", "") or "",
            "author": p.get("author"),
            "score": int(p.get("score", 0) or 0),
            "upvote_ratio": float(p.get("upvote_ratio", 0.0) or 0.0),
            "num_comments": int(p.get("num_comments", 0) or 0),
            "created_utc": float(p["created_utc"]),
            "url": p.get("url") or p.get("url_overridden_by_dest", ""),
            "is_self": bool(p.get("is_self", False)),
            "permalink": p.get("permalink", ""),
        }

    def _extract_comment_data(self, c: dict, post_id: str, subreddit: str) -> Optional[dict]:
        body = c.get("body", "")
        if not body or body in ("[deleted]", "[removed]"):
            return None
        # Filtrar respuestas automáticas de moderadores
        if c.get("distinguished") == "moderator" and len(body) < 200:
            return None

        return {
            "id": c["id"],
            "post_id": post_id,
            "subreddit": subreddit,
            "body": body,
            "author": c.get("author"),
            "score": int(c.get("score", 0) or 0),
            "created_utc": float(c["created_utc"]),
            "parent_id": c.get("parent_id", ""),
            "is_root": c.get("parent_id", "").startswith("t3_"),
            "depth": int(c.get("depth", 0) or 0),
            "controversiality": int(c.get("controversiality", 0) or 0),
        }

    def collect_historical(
        self,
        subreddit: str,
        days: int = 30,
    ) -> dict:
        """
        Recolecta posts y comentarios de los últimos N días usando Arctic Shift.

        Para cada día hace tantas requests como sean necesarias para obtener
        TODOS los posts de ese día (paginación automática).

        Args:
            subreddit: Nombre del subreddit (sin r/).
            days: Número de días hacia atrás (default: 30).

        Returns:
            dict con estadísticas y distribución diaria de posts.
        """
        logger.info(
            f"[Arctic] Iniciando recolección histórica de r/{subreddit} "
            f"({days} días via Arctic Shift API)"
        )

        run_id = self.db.start_collection_run(subreddit, {
            "mode": "arctic_historical",
            "days": days,
            "source": "arctic-shift.photon-reddit.com",
        })

        now = datetime.now(tz=timezone.utc)
        posts_seen = set()
        total_posts = 0
        new_posts = 0
        total_comments = 0
        new_comments = 0
        daily_counts = {}

        try:
            for day_offset in range(days):
                day_end = now - timedelta(days=day_offset)
                day_start = now - timedelta(days=day_offset + 1)
                epoch_start = int(day_start.timestamp())
                epoch_end = int(day_end.timestamp())
                day_label = day_start.strftime("%Y-%m-%d")

                raw_posts = self._get_posts_for_day(subreddit, epoch_start, epoch_end)
                day_count = 0

                for p in raw_posts:
                    if p["id"] in posts_seen:
                        continue
                    # Filtrar posts removidos sin contenido útil
                    selftext = p.get("selftext", "") or ""
                    title = p.get("title", "") or ""
                    if not title:
                        continue

                    posts_seen.add(p["id"])
                    post_data = self._extract_post_data(p, subreddit)
                    was_new = self.db.insert_post(post_data)
                    total_posts += 1
                    day_count += 1
                    if was_new:
                        new_posts += 1

                    # Comentarios del post
                    raw_comments = self._get_comments_for_post(p["id"])
                    comment_batch = []
                    for c in raw_comments:
                        cd = self._extract_comment_data(c, p["id"], subreddit)
                        if cd:
                            comment_batch.append(cd)

                    if comment_batch:
                        inserted = self.db.insert_comments_batch(comment_batch)
                        total_comments += len(comment_batch)
                        new_comments += inserted

                    time.sleep(REQUEST_SLEEP)

                daily_counts[day_label] = day_count
                logger.info(
                    f"  [{day_label}] {day_count} posts, "
                    f"acumulado: {total_posts} posts / {total_comments} comentarios"
                )

            self.db.finish_collection_run(run_id, new_posts, new_comments, "completed")

            days_with_data = sum(1 for c in daily_counts.values() if c > 0)
            stats = {
                "subreddit": subreddit,
                "mode": "arctic_historical",
                "days_requested": days,
                "days_with_data": days_with_data,
                "total_posts_processed": total_posts,
                "new_posts_inserted": new_posts,
                "total_comments_processed": total_comments,
                "new_comments_inserted": new_comments,
                "daily_distribution": daily_counts,
                "run_id": run_id,
            }

            logger.info(
                f"[Arctic] r/{subreddit} completado: "
                f"{new_posts} posts, {new_comments} comentarios, "
                f"{days_with_data}/{days} días con datos"
            )
            return stats

        except Exception as e:
            self.db.finish_collection_run(run_id, new_posts, new_comments, "failed")
            logger.error(f"Error en recolección Arctic de r/{subreddit}: {e}")
            raise
