"""
Script para preprocesar textos de Reddit almacenados en la base de datos.

Uso:
    python -m scripts.preprocess_data                  # Procesar lo pendiente
    python -m scripts.preprocess_data --stats          # Solo mostrar estadísticas
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import MIN_WORD_COUNT
from src.preprocessing.preprocessor import TextPreprocessor
from src.database.db_manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Preprocesar textos de Reddit")
    parser.add_argument(
        "--stats", action="store_true",
        help="Solo mostrar estadísticas sin procesar"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PREPROCESAMIENTO DE TEXTOS")
    logger.info("=" * 60)

    db = DatabaseManager()
    preprocessor = TextPreprocessor(db=db)

    if args.stats:
        stats = db.get_stats()
        logger.info(f"Posts en DB: {stats['total_posts']}")
        logger.info(f"Comentarios en DB: {stats['total_comments']}")
        logger.info(f"Textos preprocesados: {stats['total_preprocessed']}")
        logger.info(f"Textos válidos para análisis: {stats['valid_preprocessed']}")
        return

    # Procesar textos pendientes
    results = preprocessor.process_all_pending()

    logger.info("=" * 60)
    logger.info("RESUMEN DE PREPROCESAMIENTO")
    logger.info("=" * 60)
    logger.info(f"Comentarios procesados: {results['comments_processed']}")
    logger.info(f"  - Válidos (>={MIN_WORD_COUNT} palabras): {results['comments_valid']}")
    logger.info(f"  - Filtrados (muy cortos): {results['comments_filtered']}")
    logger.info(f"Posts procesados: {results['posts_processed']}")
    logger.info(f"  - Válidos: {results['posts_valid']}")
    logger.info(f"  - Filtrados: {results['posts_filtered']}")

    # Mostrar ejemplos
    logger.info("\n" + "=" * 60)
    logger.info("EJEMPLOS DE TEXTOS PREPROCESADOS")
    logger.info("=" * 60)

    sentiment_texts = preprocessor.get_texts_for_sentiment(limit=3)
    if sentiment_texts:
        logger.info("\n--- Textos para Sentimiento (RoBERTa) ---")
        for t in sentiment_texts[:3]:
            preview = t["text"][:150] + "..." if len(t["text"]) > 150 else t["text"]
            logger.info(f"  [{t['subreddit']}] {preview}")

    topic_texts = preprocessor.get_texts_for_topics(limit=3)
    if topic_texts:
        logger.info("\n--- Textos para Tópicos (BERTopic) ---")
        for t in topic_texts[:3]:
            preview = t["text"][:200] + "..." if len(t["text"]) > 200 else t["text"]
            logger.info(f"  [{t['subreddit']}] {preview}")

    # Estadísticas finales
    stats = db.get_stats()
    logger.info(f"\nEstado final de la base de datos:")
    logger.info(f"  Total textos preprocesados: {stats['total_preprocessed']}")
    logger.info(f"  Textos válidos para análisis: {stats['valid_preprocessed']}")


if __name__ == "__main__":
    main()
