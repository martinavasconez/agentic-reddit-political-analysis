"""
Módulo de recolección de datos de Reddit.
Extrae posts y comentarios de subreddits políticos de forma on-demand.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import praw
from loguru import logger

from config.settings import (
    POSTS_PER_SUBREDDIT,
    COMMENTS_PER_POST,
    RATE_LIMIT_SLEEP,
    DEFAULT_COLLECTION_DAYS,
    TARGET_SUBREDDITS,
)
from src.collection.reddit_client import create_reddit_client
from src.database.db_manager import DatabaseManager


class RedditCollector:
    """Recolector de datos de Reddit para análisis político."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.reddit = create_reddit_client()
        self.db = db or DatabaseManager()

    def collect_subreddit(
        self,
        subreddit_name: str,
        days: int = DEFAULT_COLLECTION_DAYS,
        max_posts: int = POSTS_PER_SUBREDDIT,
        max_comments_per_post: int = COMMENTS_PER_POST,
        cutoff_minutes: Optional[int] = None,
    ) -> dict:
        """
        Recolecta posts y comentarios de un subreddit.

        Args:
            subreddit_name: Nombre del subreddit (sin r/).
            days: Número de días hacia atrás para recolectar.
            max_posts: Máximo de posts a recolectar.
            max_comments_per_post: Máximo de comentarios por post.
            cutoff_minutes: Si se especifica, sobreescribe days con N minutos.

        Returns:
            dict con estadísticas de la recolección.
        """
        if cutoff_minutes is not None:
            logger.info(f"Iniciando recolección de r/{subreddit_name} (últimos {cutoff_minutes} minutos)")
            cutoff_timestamp = (datetime.utcnow() - timedelta(minutes=cutoff_minutes)).timestamp()
        else:
            logger.info(f"Iniciando recolección de r/{subreddit_name} (últimos {days} días)")
            cutoff_timestamp = (datetime.utcnow() - timedelta(days=days)).timestamp()

        # Registrar run en la base de datos
        run_id = self.db.start_collection_run(subreddit_name, {
            "days": days,
            "max_posts": max_posts,
            "max_comments_per_post": max_comments_per_post,
        })

        total_posts = 0
        total_comments = 0
        new_posts = 0
        new_comments = 0

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Recolectar posts usando "new" para tener los más recientes
            # y "top" del período para tener los más relevantes
            posts_collected = set()

            for sort_method in ["new", "hot", "top"]:
                if len(posts_collected) >= max_posts:
                    break

                logger.info(f"  Recolectando posts ({sort_method}) de r/{subreddit_name}...")

                if sort_method == "new":
                    posts_iter = subreddit.new(limit=max_posts)
                elif sort_method == "hot":
                    posts_iter = subreddit.hot(limit=max_posts)
                else:
                    posts_iter = subreddit.top(time_filter="month", limit=max_posts)

                for post in posts_iter:
                    if len(posts_collected) >= max_posts:
                        break

                    # Filtrar por fecha
                    if post.created_utc < cutoff_timestamp:
                        if sort_method == "new":
                            break  # Si es "new" y ya pasamos la fecha, no hay más
                        continue

                    # Evitar duplicados dentro de esta misma ejecución
                    if post.id in posts_collected:
                        continue
                    posts_collected.add(post.id)

                    # Extraer datos del post
                    post_data = self._extract_post_data(post)
                    was_new = self.db.insert_post(post_data)
                    total_posts += 1
                    if was_new:
                        new_posts += 1

                    # Recolectar comentarios del post
                    comment_count = self._collect_comments(
                        post, subreddit_name, max_comments_per_post
                    )
                    total_comments += comment_count["total"]
                    new_comments += comment_count["new"]

                    # Respetar rate limits
                    time.sleep(RATE_LIMIT_SLEEP)

                logger.info(
                    f"  {sort_method}: {len(posts_collected)} posts acumulados"
                )

            # Finalizar run
            self.db.finish_collection_run(run_id, new_posts, new_comments, "completed")

            stats = {
                "subreddit": subreddit_name,
                "total_posts_processed": total_posts,
                "new_posts_inserted": new_posts,
                "total_comments_processed": total_comments,
                "new_comments_inserted": new_comments,
                "run_id": run_id,
            }

            logger.info(
                f"Recolección de r/{subreddit_name} completada: "
                f"{new_posts} posts nuevos, {new_comments} comentarios nuevos"
            )
            return stats

        except Exception as e:
            self.db.finish_collection_run(run_id, new_posts, new_comments, "failed")
            logger.error(f"Error en recolección de r/{subreddit_name}: {e}")
            raise

    def collect_historical(
        self,
        subreddit_name: str,
        days: int = 30,
        max_comments_per_post: int = COMMENTS_PER_POST,
    ) -> dict:
        """
        Recolecta TODOS los posts disponibles distribuidos a lo largo de N días
        usando búsqueda con rango de timestamps (Lucene syntax de Reddit).

        Estrategia:
          1. Para cada día de los últimos N días: busca TODOS los posts con
             timestamp:EPOCH_START..EPOCH_END (sin límite por día, Reddit
             devuelve hasta ~250 resultados por query con paginación).
          2. Complementa con top(time_filter="month") para capturar posts
             relevantes que la búsqueda por día pueda perder.

        Esto garantiza distribución temporal diaria, necesaria para el
        cálculo de Δ del Agente de Tendencias.

        Args:
            subreddit_name: Nombre del subreddit (sin r/).
            days: Número de días hacia atrás (default: 30).
            max_comments_per_post: Comentarios máximos por post.

        Returns:
            dict con estadísticas y distribución diaria de posts.
        """
        logger.info(
            f"[Histórico] Iniciando recolección de r/{subreddit_name} "
            f"({days} días, sin límite de posts/día)"
        )

        run_id = self.db.start_collection_run(subreddit_name, {
            "mode": "historical",
            "days": days,
            "max_comments_per_post": max_comments_per_post,
        })

        now = datetime.utcnow()
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_seen = set()
        total_posts = 0
        new_posts = 0
        total_comments = 0
        new_comments = 0
        daily_counts = {}

        try:
            # Fase 1: búsqueda día por día con timestamp Lucene
            for day_offset in range(days):
                day_end = now - timedelta(days=day_offset)
                day_start = now - timedelta(days=day_offset + 1)
                epoch_start = int(day_start.timestamp())
                epoch_end = int(day_end.timestamp())
                day_label = day_start.strftime("%Y-%m-%d")

                query = f"timestamp:{epoch_start}..{epoch_end}"
                day_count = 0

                try:
                    results = subreddit.search(
                        query=query,
                        sort="top",
                        syntax="lucene",
                        limit=None,      # sin límite — PRAW pagina hasta agotar resultados
                        time_filter="all",
                    )

                    for post in results:
                        if post.id in posts_seen:
                            continue
                        posts_seen.add(post.id)

                        post_data = self._extract_post_data(post)
                        was_new = self.db.insert_post(post_data)
                        total_posts += 1
                        day_count += 1
                        if was_new:
                            new_posts += 1

                        comment_count = self._collect_comments(
                            post, subreddit_name, max_comments_per_post
                        )
                        total_comments += comment_count["total"]
                        new_comments += comment_count["new"]

                        time.sleep(RATE_LIMIT_SLEEP)

                except Exception as e:
                    logger.warning(f"  Error en búsqueda día {day_label}: {e}")

                daily_counts[day_label] = day_count
                logger.info(
                    f"  [{day_label}] {day_count} posts "
                    f"(acumulado: {total_posts})"
                )
                time.sleep(RATE_LIMIT_SLEEP)

            # Fase 2: top del mes para complementar posts relevantes
            logger.info(f"  Complementando con top(month) de r/{subreddit_name}...")
            try:
                for post in subreddit.top(time_filter="month", limit=500):
                    if post.id in posts_seen:
                        continue
                    cutoff_ts = (now - timedelta(days=days)).timestamp()
                    if post.created_utc < cutoff_ts:
                        continue
                    posts_seen.add(post.id)

                    post_data = self._extract_post_data(post)
                    was_new = self.db.insert_post(post_data)
                    total_posts += 1
                    if was_new:
                        new_posts += 1

                    comment_count = self._collect_comments(
                        post, subreddit_name, max_comments_per_post
                    )
                    total_comments += comment_count["total"]
                    new_comments += comment_count["new"]

                    time.sleep(RATE_LIMIT_SLEEP)

            except Exception as e:
                logger.warning(f"  Error en top(month): {e}")

            self.db.finish_collection_run(run_id, new_posts, new_comments, "completed")

            stats = {
                "subreddit": subreddit_name,
                "mode": "historical",
                "days_requested": days,
                "total_posts_processed": total_posts,
                "new_posts_inserted": new_posts,
                "total_comments_processed": total_comments,
                "new_comments_inserted": new_comments,
                "daily_distribution": daily_counts,
                "run_id": run_id,
            }

            days_with_data = sum(1 for c in daily_counts.values() if c > 0)
            logger.info(
                f"[Histórico] r/{subreddit_name} completado: "
                f"{new_posts} posts, {new_comments} comentarios, "
                f"{days_with_data}/{days} días con datos"
            )
            return stats

        except Exception as e:
            self.db.finish_collection_run(run_id, new_posts, new_comments, "failed")
            logger.error(f"Error en recolección histórica de r/{subreddit_name}: {e}")
            raise

    def collect_all(
        self,
        days: int = DEFAULT_COLLECTION_DAYS,
        subreddits: Optional[list[str]] = None,
        cutoff_minutes: Optional[int] = None,
    ) -> list[dict]:
        """
        Recolecta datos de todos los subreddits objetivo.

        Args:
            days: Número de días hacia atrás.
            subreddits: Lista de subreddits (usa TARGET_SUBREDDITS si None).
            cutoff_minutes: Si se especifica, sobreescribe days con N minutos.

        Returns:
            Lista de estadísticas por subreddit.
        """
        targets = subreddits or TARGET_SUBREDDITS
        all_stats = []

        for sub in targets:
            try:
                stats = self.collect_subreddit(sub, days=days, cutoff_minutes=cutoff_minutes)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Falló recolección de r/{sub}: {e}")
                all_stats.append({"subreddit": sub, "error": str(e)})

        # Resumen
        total_new_posts = sum(s.get("new_posts_inserted", 0) for s in all_stats)
        total_new_comments = sum(s.get("new_comments_inserted", 0) for s in all_stats)
        logger.info(
            f"Recolección total: {total_new_posts} posts nuevos, "
            f"{total_new_comments} comentarios nuevos"
        )

        return all_stats

    def _extract_post_data(self, post: praw.models.Submission) -> dict:
        """Extrae los campos relevantes de un post de Reddit."""
        return {
            "id": post.id,
            "subreddit": post.subreddit.display_name,
            "title": post.title,
            "selftext": post.selftext if post.is_self else "",
            "author": str(post.author) if post.author else None,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc,
            "url": post.url,
            "is_self": post.is_self,
            "permalink": post.permalink,
        }

    def _collect_comments(
        self,
        post: praw.models.Submission,
        subreddit_name: str,
        max_comments: int,
    ) -> dict:
        """
        Recolecta comentarios de un post.

        Returns:
            dict con "total" y "new" (cantidad de comentarios nuevos insertados).
        """
        total = 0
        new = 0

        try:
            # replace_more(limit=0) descarta los "load more comments" para evitar
            # requests extras excesivos. Esto obtiene los comentarios ya cargados.
            post.comments.replace_more(limit=0)
            comments = post.comments.list()[:max_comments]

            batch = []
            for comment in comments:
                # Filtrar comentarios eliminados o sin cuerpo
                if not hasattr(comment, "body"):
                    continue
                if comment.body in ("[deleted]", "[removed]"):
                    continue

                comment_data = {
                    "id": comment.id,
                    "post_id": post.id,
                    "subreddit": subreddit_name,
                    "body": comment.body,
                    "author": str(comment.author) if comment.author else None,
                    "score": comment.score,
                    "created_utc": comment.created_utc,
                    "parent_id": comment.parent_id,
                    "is_root": comment.is_root,
                    "depth": getattr(comment, "depth", 0),
                    "controversiality": getattr(comment, "controversiality", 0),
                }
                batch.append(comment_data)
                total += 1

            if batch:
                new = self.db.insert_comments_batch(batch)

        except Exception as e:
            logger.warning(f"Error recolectando comentarios del post {post.id}: {e}")

        return {"total": total, "new": new}
