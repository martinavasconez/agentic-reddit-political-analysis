"""
Pipeline de preprocesamiento de textos de Reddit.
Procesa posts y comentarios almacenados en la base de datos,
generando versiones limpias optimizadas para cada agente.
"""

from typing import Optional

from loguru import logger

from config.settings import MIN_WORD_COUNT, BERTOPIC_MIN_WORDS, MAX_TEXT_LENGTH
from src.database.db_manager import DatabaseManager
from src.preprocessing.text_cleaner import TextCleaner


class TextPreprocessor:
    """Pipeline de preprocesamiento de textos de Reddit."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.cleaner = TextCleaner()

    def process_comment(self, comment: dict) -> Optional[dict]:
        """
        Procesa un comentario individual.

        Args:
            comment: dict con los campos del comentario (de la tabla comments).

        Returns:
            dict con texto preprocesado o None si no pasa los filtros.
        """
        original_text = comment.get("body", "")

        # Filtro: texto vacío o eliminado — insertar is_valid=0 para marcar como procesado
        # y evitar que vuelva a aparecer como pendiente en el loop
        if not original_text or not original_text.strip() or original_text in ("[deleted]", "[removed]"):
            return {
                "source_id": comment["id"],
                "source_type": "comment",
                "subreddit": comment["subreddit"],
                "original_text": original_text or "",
                "cleaned_text": "",
                "text_for_sentiment": "",
                "text_for_topics": "",
                "word_count": 0,
                "created_utc": comment["created_utc"],
                "is_valid": False,
            }

        def _invalid(reason_text):
            return {
                "source_id": comment["id"],
                "source_type": "comment",
                "subreddit": comment["subreddit"],
                "original_text": reason_text,
                "cleaned_text": "",
                "text_for_sentiment": "",
                "text_for_topics": "",
                "word_count": 0,
                "created_utc": comment["created_utc"],
                "is_valid": False,
            }

        # Filtro: contenido de bots
        if self.cleaner.is_bot_content(original_text):
            return _invalid(original_text)

        # Filtro: texto demasiado largo (posible spam)
        if len(original_text) > MAX_TEXT_LENGTH:
            original_text = original_text[:MAX_TEXT_LENGTH]

        # Limpieza base
        cleaned = self.cleaner.clean_base(original_text)
        if not cleaned:
            return _invalid(original_text)

        # Versiones específicas para cada agente
        text_sentiment = self.cleaner.clean_for_sentiment(original_text)
        text_topics = self.cleaner.clean_for_topics(original_text)

        # Conteo de palabras sobre el texto limpio
        wc = self.cleaner.word_count(cleaned)

        # Filtro: comentarios muy cortos
        is_valid = wc >= MIN_WORD_COUNT

        return {
            "source_id": comment["id"],
            "source_type": "comment",
            "subreddit": comment["subreddit"],
            "original_text": original_text,
            "cleaned_text": cleaned,
            "text_for_sentiment": text_sentiment,
            "text_for_topics": text_topics,
            "word_count": wc,
            "created_utc": comment["created_utc"],
            "is_valid": is_valid,
        }

    def process_post(self, post: dict) -> Optional[dict]:
        """
        Procesa un post individual (solo self posts con texto).

        Para posts, combinamos título + selftext como texto completo,
        ya que el título contiene información contextual importante.

        Args:
            post: dict con los campos del post (de la tabla posts).

        Returns:
            dict con texto preprocesado o None si no pasa los filtros.
        """
        title = post.get("title", "")
        selftext = post.get("selftext", "")

        # Para posts, el texto relevante es título + cuerpo
        if selftext and selftext not in ("[deleted]", "[removed]"):
            original_text = f"{title}. {selftext}"
        else:
            original_text = title

        if not original_text or not original_text.strip():
            return None

        if len(original_text) > MAX_TEXT_LENGTH:
            original_text = original_text[:MAX_TEXT_LENGTH]

        # Limpieza
        cleaned = self.cleaner.clean_base(original_text)
        if not cleaned:
            return None

        text_sentiment = self.cleaner.clean_for_sentiment(original_text)
        text_topics = self.cleaner.clean_for_topics(original_text)

        wc = self.cleaner.word_count(cleaned)
        is_valid = wc >= MIN_WORD_COUNT

        return {
            "source_id": post["id"],
            "source_type": "post",
            "subreddit": post["subreddit"],
            "original_text": original_text,
            "cleaned_text": cleaned,
            "text_for_sentiment": text_sentiment,
            "text_for_topics": text_topics,
            "word_count": wc,
            "created_utc": post["created_utc"],
            "is_valid": is_valid,
        }

    def process_all_pending(self) -> dict:
        """
        Procesa todos los comentarios y posts que aún no han sido preprocesados.

        Returns:
            dict con estadísticas del procesamiento.
        """
        stats = {
            "comments_processed": 0,
            "comments_valid": 0,
            "comments_filtered": 0,
            "posts_processed": 0,
            "posts_valid": 0,
            "posts_filtered": 0,
        }

        # Procesar comentarios pendientes
        unprocessed_comments = self.db.get_unprocessed_comments()
        logger.info(f"Comentarios pendientes de preprocesar: {len(unprocessed_comments)}")

        if unprocessed_comments:
            batch = []
            for comment in unprocessed_comments:
                result = self.process_comment(comment)
                if result:
                    batch.append(result)
                    stats["comments_processed"] += 1
                    if result["is_valid"]:
                        stats["comments_valid"] += 1
                    else:
                        stats["comments_filtered"] += 1

            if batch:
                self.db.insert_preprocessed_batch(batch)
                logger.info(
                    f"Comentarios preprocesados: {stats['comments_processed']} "
                    f"(válidos: {stats['comments_valid']}, "
                    f"filtrados por longitud: {stats['comments_filtered']})"
                )

        # Procesar posts pendientes
        unprocessed_posts = self.db.get_unprocessed_posts()
        logger.info(f"Posts pendientes de preprocesar: {len(unprocessed_posts)}")

        if unprocessed_posts:
            batch = []
            for post in unprocessed_posts:
                result = self.process_post(post)
                if result:
                    batch.append(result)
                    stats["posts_processed"] += 1
                    if result["is_valid"]:
                        stats["posts_valid"] += 1
                    else:
                        stats["posts_filtered"] += 1

            if batch:
                self.db.insert_preprocessed_batch(batch)
                logger.info(
                    f"Posts preprocesados: {stats['posts_processed']} "
                    f"(válidos: {stats['posts_valid']}, "
                    f"filtrados por longitud: {stats['posts_filtered']})"
                )

        # Estadísticas generales
        total = stats["comments_valid"] + stats["posts_valid"]
        logger.info(f"Total de textos válidos para análisis: {total}")

        return stats

    def get_texts_for_sentiment(self, subreddit: Optional[str] = None,
                                 limit: int = 5000) -> list[dict]:
        """Obtiene textos preprocesados para análisis de sentimiento."""
        rows = self.db.get_preprocessed_texts(
            subreddit=subreddit, valid_only=True, limit=limit
        )
        return [
            {
                "source_id": r["source_id"],
                "source_type": r["source_type"],
                "subreddit": r["subreddit"],
                "text": r["text_for_sentiment"],
                "created_utc": r["created_utc"],
            }
            for r in rows
            if r["text_for_sentiment"]
        ]

    def get_texts_for_topics(self, subreddit: Optional[str] = None,
                              limit: int = 5000) -> list[dict]:
        """
        Obtiene textos preprocesados para detección de tópicos.
        Filtra adicionalmente por longitud mínima para BERTopic.
        """
        rows = self.db.get_preprocessed_texts(
            subreddit=subreddit, valid_only=True, limit=limit
        )
        return [
            {
                "source_id": r["source_id"],
                "source_type": r["source_type"],
                "subreddit": r["subreddit"],
                "text": r["text_for_topics"],
                "created_utc": r["created_utc"],
            }
            for r in rows
            if r["text_for_topics"] and r["word_count"] >= BERTOPIC_MIN_WORDS
        ]
