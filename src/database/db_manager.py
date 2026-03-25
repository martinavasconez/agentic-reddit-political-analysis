"""
Gestión de la base de datos SQLite.
Operaciones CRUD para posts, comentarios y textos preprocesados.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional
from loguru import logger

from src.database.models import SCHEMA_SQL
from config.settings import DB_PATH


class DatabaseManager:
    """Gestiona conexiones y operaciones sobre la base de datos SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Crea las tablas si no existen."""
        conn = self._get_connection()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            logger.info(f"Base de datos inicializada en {self.db_path}")
        finally:
            conn.close()

    # Posts

    def insert_post(self, post_data: dict) -> bool:
        """Inserta un post. Retorna True si se insertó (nuevo), False si ya existía."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO posts
                (id, subreddit, title, selftext, author, score, upvote_ratio,
                 num_comments, created_utc, url, is_self, permalink, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post_data["id"],
                post_data["subreddit"],
                post_data["title"],
                post_data.get("selftext", ""),
                post_data.get("author"),
                post_data.get("score", 0),
                post_data.get("upvote_ratio", 0.0),
                post_data.get("num_comments", 0),
                post_data["created_utc"],
                post_data.get("url"),
                int(post_data.get("is_self", False)),
                post_data.get("permalink"),
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def insert_posts_batch(self, posts: list[dict]) -> int:
        """Inserta múltiples posts. Retorna cantidad de nuevos insertados."""
        conn = self._get_connection()
        inserted = 0
        try:
            for post_data in posts:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO posts
                    (id, subreddit, title, selftext, author, score, upvote_ratio,
                     num_comments, created_utc, url, is_self, permalink, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_data["id"],
                    post_data["subreddit"],
                    post_data["title"],
                    post_data.get("selftext", ""),
                    post_data.get("author"),
                    post_data.get("score", 0),
                    post_data.get("upvote_ratio", 0.0),
                    post_data.get("num_comments", 0),
                    post_data["created_utc"],
                    post_data.get("url"),
                    int(post_data.get("is_self", False)),
                    post_data.get("permalink"),
                    datetime.utcnow().isoformat(),
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            conn.commit()
            return inserted
        finally:
            conn.close()

    def post_exists(self, post_id: str) -> bool:
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT 1 FROM posts WHERE id = ?", (post_id,)).fetchone()
            return row is not None
        finally:
            conn.close()

    def get_posts(self, subreddit: Optional[str] = None, limit: int = 100) -> list[dict]:
        conn = self._get_connection()
        try:
            if subreddit:
                rows = conn.execute(
                    "SELECT * FROM posts WHERE subreddit = ? ORDER BY created_utc DESC LIMIT ?",
                    (subreddit, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM posts ORDER BY created_utc DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # Comments

    def insert_comment(self, comment_data: dict) -> bool:
        """Inserta un comentario. Retorna True si se insertó (nuevo)."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO comments
                (id, post_id, subreddit, body, author, score, created_utc,
                 parent_id, is_root, depth, controversiality, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comment_data["id"],
                comment_data["post_id"],
                comment_data["subreddit"],
                comment_data["body"],
                comment_data.get("author"),
                comment_data.get("score", 0),
                comment_data["created_utc"],
                comment_data.get("parent_id"),
                int(comment_data.get("is_root", False)),
                comment_data.get("depth", 0),
                comment_data.get("controversiality", 0),
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def insert_comments_batch(self, comments: list[dict]) -> int:
        """Inserta múltiples comentarios. Retorna cantidad de nuevos insertados."""
        conn = self._get_connection()
        inserted = 0
        try:
            for c in comments:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO comments
                    (id, post_id, subreddit, body, author, score, created_utc,
                     parent_id, is_root, depth, controversiality, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c["id"],
                    c["post_id"],
                    c["subreddit"],
                    c["body"],
                    c.get("author"),
                    c.get("score", 0),
                    c["created_utc"],
                    c.get("parent_id"),
                    int(c.get("is_root", False)),
                    c.get("depth", 0),
                    c.get("controversiality", 0),
                    datetime.utcnow().isoformat(),
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            conn.commit()
            return inserted
        finally:
            conn.close()

    def get_comments(self, post_id: Optional[str] = None, subreddit: Optional[str] = None,
                     limit: int = 1000) -> list[dict]:
        conn = self._get_connection()
        try:
            if post_id:
                rows = conn.execute(
                    "SELECT * FROM comments WHERE post_id = ? ORDER BY score DESC LIMIT ?",
                    (post_id, limit)
                ).fetchall()
            elif subreddit:
                rows = conn.execute(
                    "SELECT * FROM comments WHERE subreddit = ? ORDER BY created_utc DESC LIMIT ?",
                    (subreddit, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM comments ORDER BY created_utc DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # Preprocessed Texts

    def insert_preprocessed_text(self, text_data: dict) -> bool:
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO preprocessed_texts
                (source_id, source_type, subreddit, original_text, cleaned_text,
                 text_for_sentiment, text_for_topics, word_count, created_utc, processed_at, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text_data["source_id"],
                text_data["source_type"],
                text_data["subreddit"],
                text_data["original_text"],
                text_data["cleaned_text"],
                text_data["text_for_sentiment"],
                text_data["text_for_topics"],
                text_data["word_count"],
                text_data["created_utc"],
                datetime.utcnow().isoformat(),
                int(text_data.get("is_valid", True)),
            ))
            conn.commit()
            return True
        finally:
            conn.close()

    def insert_preprocessed_batch(self, texts: list[dict]) -> int:
        conn = self._get_connection()
        inserted = 0
        try:
            for t in texts:
                conn.execute("""
                    INSERT OR REPLACE INTO preprocessed_texts
                    (source_id, source_type, subreddit, original_text, cleaned_text,
                     text_for_sentiment, text_for_topics, word_count, created_utc, processed_at, is_valid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t["source_id"],
                    t["source_type"],
                    t["subreddit"],
                    t["original_text"],
                    t["cleaned_text"],
                    t["text_for_sentiment"],
                    t["text_for_topics"],
                    t["word_count"],
                    t["created_utc"],
                    datetime.utcnow().isoformat(),
                    int(t.get("is_valid", True)),
                ))
                inserted += 1
            conn.commit()
            return inserted
        finally:
            conn.close()

    def get_preprocessed_texts(self, subreddit: Optional[str] = None,
                               source_type: Optional[str] = None,
                               valid_only: bool = True,
                               limit: int = 1000) -> list[dict]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM preprocessed_texts WHERE 1=1"
            params = []
            if valid_only:
                query += " AND is_valid = 1"
            if subreddit:
                query += " AND subreddit = ?"
                params.append(subreddit)
            if source_type:
                query += " AND source_type = ?"
                params.append(source_type)
            query += " ORDER BY created_utc DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_unprocessed_comments(self, limit: int = 5000) -> list[dict]:
        """Obtiene comentarios que aún no han sido preprocesados."""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT c.* FROM comments c
                LEFT JOIN preprocessed_texts pt
                    ON c.id = pt.source_id AND pt.source_type = 'comment'
                WHERE pt.id IS NULL
                  AND c.body NOT IN ('[deleted]', '[removed]')
                ORDER BY c.created_utc DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_unprocessed_posts(self, limit: int = 5000) -> list[dict]:
        """Obtiene posts (self posts con texto) que aún no han sido preprocesados."""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT p.* FROM posts p
                LEFT JOIN preprocessed_texts pt
                    ON p.id = pt.source_id AND pt.source_type = 'post'
                WHERE pt.id IS NULL
                  AND p.is_self = 1
                  AND p.selftext != ''
                  AND p.selftext NOT IN ('[deleted]', '[removed]')
                ORDER BY p.created_utc DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # Collection Runs

    def start_collection_run(self, subreddit: str, parameters: dict) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO collection_runs (subreddit, started_at, parameters)
                VALUES (?, ?, ?)
            """, (subreddit, datetime.utcnow().isoformat(), json.dumps(parameters)))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def finish_collection_run(self, run_id: int, posts: int, comments: int,
                               status: str = "completed"):
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE collection_runs
                SET finished_at = ?, posts_collected = ?, comments_collected = ?, status = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), posts, comments, status, run_id))
            conn.commit()
        finally:
            conn.close()

    # Sentiment Results

    def get_unanalyzed_texts_for_sentiment(self, limit: int = 1000) -> list[dict]:
        """Textos preprocesados válidos que aún no tienen resultado de sentimiento."""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT pt.* FROM preprocessed_texts pt
                LEFT JOIN sentiment_results sr
                    ON pt.source_id = sr.source_id AND pt.source_type = sr.source_type
                WHERE pt.is_valid = 1
                  AND sr.id IS NULL
                ORDER BY pt.created_utc DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def insert_sentiment_batch(self, results: list[dict]) -> int:
        """Inserta resultados de sentimiento. Retorna cantidad insertada."""
        conn = self._get_connection()
        inserted = 0
        try:
            for r in results:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO sentiment_results
                    (preprocessed_text_id, source_id, source_type, subreddit,
                     roberta_label, roberta_confidence, decision,
                     final_label, final_confidence,
                     vader_compound, vader_label, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r["preprocessed_text_id"],
                    r["source_id"],
                    r["source_type"],
                    r["subreddit"],
                    r["roberta_label"],
                    r["roberta_confidence"],
                    r["decision"],
                    r["final_label"],
                    r["final_confidence"],
                    r.get("vader_compound"),
                    r.get("vader_label"),
                    r["analyzed_at"],
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            conn.commit()
            return inserted
        finally:
            conn.close()

    def get_sentiment_stats(self, subreddit: Optional[str] = None) -> dict:
        """Métricas agregadas de sentimiento."""
        conn = self._get_connection()
        try:
            base = "WHERE 1=1"
            params = []
            if subreddit:
                base += " AND subreddit = ?"
                params.append(subreddit)

            total = conn.execute(
                f"SELECT COUNT(*) FROM sentiment_results {base}", params
            ).fetchone()[0]

            rows = conn.execute(
                f"SELECT final_label, COUNT(*) as cnt FROM sentiment_results {base} GROUP BY final_label",
                params
            ).fetchall()
            dist = {r["final_label"]: r["cnt"] for r in rows}

            rows_dec = conn.execute(
                f"SELECT decision, COUNT(*) as cnt FROM sentiment_results {base} GROUP BY decision",
                params
            ).fetchall()
            decisions = {r["decision"]: r["cnt"] for r in rows_dec}

            avg_conf = conn.execute(
                f"SELECT AVG(final_confidence) FROM sentiment_results {base} AND decision != 'ambiguous'",
                params
            ).fetchone()[0]

            return {
                "total_analyzed": total,
                "label_distribution": dist,
                "decision_distribution": decisions,
                "avg_confidence": round(avg_conf or 0, 4),
                "pct_ambiguous": round(dist.get("ambiguous", 0) / total * 100, 2) if total else 0,
            }
        finally:
            conn.close()

    # Topic Assignments

    def get_texts_for_topic_modeling(self, limit: int = 50000) -> list[dict]:
        """Textos válidos con su timestamp para modelado de tópicos."""
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT id, source_id, source_type, subreddit,
                       text_for_topics, created_utc
                FROM preprocessed_texts
                WHERE is_valid = 1
                ORDER BY created_utc ASC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def insert_topic_assignments_batch(self, assignments: list[dict]) -> int:
        conn = self._get_connection()
        inserted = 0
        try:
            for a in assignments:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO topic_assignments
                    (preprocessed_text_id, source_id, source_type, subreddit,
                     created_utc, topic_id, topic_label, topic_probability,
                     model_run_id, assigned_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    a["preprocessed_text_id"],
                    a["source_id"],
                    a["source_type"],
                    a["subreddit"],
                    a["created_utc"],
                    a["topic_id"],
                    a.get("topic_label"),
                    a.get("topic_probability"),
                    a["model_run_id"],
                    a["assigned_at"],
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            conn.commit()
            return inserted
        finally:
            conn.close()

    def insert_trend_analysis_batch(self, trends: list[dict]) -> int:
        conn = self._get_connection()
        inserted = 0
        try:
            for t in trends:
                cursor = conn.execute("""
                    INSERT INTO trend_analysis
                    (model_run_id, topic_id, topic_label,
                     current_weight, historical_mean, historical_std, effective_std, delta,
                     current_window_start, current_window_end,
                     historical_window_start, historical_window_end,
                     n_current_texts, n_historical_texts, corpus_coverage,
                     consecutive_growth_days, trend_decision, trend_reason,
                     daily_weights_json, analyzed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t["model_run_id"],
                    t["topic_id"],
                    t.get("topic_label"),
                    t["current_weight"],
                    t["historical_mean"],
                    t["historical_std"],
                    t["effective_std"],
                    t["delta"],
                    t["current_window_start"],
                    t["current_window_end"],
                    t["historical_window_start"],
                    t["historical_window_end"],
                    t["n_current_texts"],
                    t["n_historical_texts"],
                    t["corpus_coverage"],
                    t.get("consecutive_growth_days", 0),
                    t["trend_decision"],
                    t["trend_reason"],
                    t.get("daily_weights_json"),
                    t["analyzed_at"],
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            conn.commit()
            return inserted
        finally:
            conn.close()

    def get_trend_results(self, model_run_id: str,
                          decision_filter: Optional[str] = None) -> list[dict]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM trend_analysis WHERE model_run_id = ?"
            params = [model_run_id]
            if decision_filter:
                query += " AND trend_decision = ?"
                params.append(decision_filter)
            query += " ORDER BY delta DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_latest_topic_model_run(self) -> Optional[str]:
        """Retorna el model_run_id más reciente en topic_assignments."""
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT model_run_id FROM topic_assignments
                ORDER BY assigned_at DESC LIMIT 1
            """).fetchone()
            return row["model_run_id"] if row else None
        finally:
            conn.close()

    # Estadísticas

    def get_stats(self) -> dict:
        conn = self._get_connection()
        try:
            stats = {}
            stats["total_posts"] = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            stats["total_comments"] = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
            stats["total_preprocessed"] = conn.execute(
                "SELECT COUNT(*) FROM preprocessed_texts"
            ).fetchone()[0]
            stats["valid_preprocessed"] = conn.execute(
                "SELECT COUNT(*) FROM preprocessed_texts WHERE is_valid = 1"
            ).fetchone()[0]

            # Por subreddit
            rows = conn.execute("""
                SELECT subreddit, COUNT(*) as count FROM posts GROUP BY subreddit
            """).fetchall()
            stats["posts_by_subreddit"] = {r["subreddit"]: r["count"] for r in rows}

            rows = conn.execute("""
                SELECT subreddit, COUNT(*) as count FROM comments GROUP BY subreddit
            """).fetchall()
            stats["comments_by_subreddit"] = {r["subreddit"]: r["count"] for r in rows}

            return stats
        finally:
            conn.close()
