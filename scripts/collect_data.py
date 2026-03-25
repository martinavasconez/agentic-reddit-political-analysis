"""
Script para recolectar datos de Reddit.

Uso:
    python -m scripts.collect_data                        # Últimos 7 días vía PRAW (una vez)
    python -m scripts.collect_data --days 7               # Últimos N días vía PRAW
    python -m scripts.collect_data --arctic --days 82     # 82 días históricos vía Arctic Shift
    python -m scripts.collect_data --continuous           # Bucle continuo (Ctrl+C para parar)
    python -m scripts.collect_data --continuous --interval 3600  # Bucle cada hora
    python -m scripts.collect_data --live --minutes 5     # Demo: últimos 5 minutos + preprocesar
"""

import argparse
import sys
import time
from pathlib import Path

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import DEFAULT_COLLECTION_DAYS, TARGET_SUBREDDITS
from src.collection.collector import RedditCollector
from src.database.db_manager import DatabaseManager


def show_summary(results, db):
    """Muestra resumen de una recolección."""
    for r in results:
        if "error" in r:
            logger.error(f"  r/{r['subreddit']}: ERROR - {r['error']}")
        else:
            logger.info(
                f"  r/{r['subreddit']}: "
                f"{r['new_posts_inserted']} posts nuevos, "
                f"{r['new_comments_inserted']} comentarios nuevos"
            )

    stats = db.get_stats()
    logger.info(f"  BD total: {stats['total_posts']} posts, {stats['total_comments']} comentarios")


def run_once(collector, db, args):
    """Ejecución única de recolección."""
    logger.info("=" * 60)
    logger.info("RECOLECCIÓN DE DATOS DE REDDIT")
    logger.info("=" * 60)

    subreddits = args.subreddits or TARGET_SUBREDDITS
    logger.info(f"Subreddits: {subreddits}")
    logger.info(f"Período: últimos {args.days} días")

    results = collector.collect_all(days=args.days, subreddits=subreddits)
    show_summary(results, db)
    return results


def run_continuous(collector, db, args):
    """Ejecución en bucle continuo."""
    interval = args.interval
    subreddits = args.subreddits or TARGET_SUBREDDITS

    logger.info("=" * 60)
    logger.info("RECOLECCIÓN CONTINUA DE REDDIT")
    logger.info(f"Intervalo: {interval} segundos | Ctrl+C para detener")
    logger.info("=" * 60)

    iteration = 0
    total_new_posts = 0
    total_new_comments = 0

    try:
        while True:
            iteration += 1
            logger.info(f"\n--- Iteración {iteration} ---")

            results = collector.collect_all(days=args.days, subreddits=subreddits)

            # Acumular totales
            for r in results:
                if "error" not in r:
                    total_new_posts += r.get("new_posts_inserted", 0)
                    total_new_comments += r.get("new_comments_inserted", 0)

            show_summary(results, db)
            logger.info(
                f"  Acumulado total: {total_new_posts} posts, "
                f"{total_new_comments} comentarios"
            )

            logger.info(f"Esperando {interval} segundos...")
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("RECOLECCIÓN CONTINUA DETENIDA")
        logger.info(f"Iteraciones completadas: {iteration}")
        logger.info(f"Total recolectado: {total_new_posts} posts, {total_new_comments} comentarios")

        stats = db.get_stats()
        logger.info(f"BD final: {stats['total_posts']} posts, {stats['total_comments']} comentarios")
        logger.info("=" * 60)


def run_live_demo(collector, db, args):
    """Demo en vivo: recolecta últimos N minutos y preprocesa."""
    from src.preprocessing.preprocessor import TextPreprocessor

    minutes = args.minutes
    subreddits = args.subreddits or TARGET_SUBREDDITS

    logger.info("=" * 60)
    logger.info(f"DEMO EN VIVO — Últimos {minutes} minutos")
    logger.info("=" * 60)

    # Paso 1: Recolectar
    logger.info(f"\n[1/3] Recolectando posts de los últimos {minutes} minutos...")
    results = collector.collect_all(
        days=1, subreddits=subreddits, cutoff_minutes=minutes
    )
    show_summary(results, db)

    # Paso 2: Preprocesar
    logger.info(f"\n[2/3] Preprocesando textos...")
    preprocessor = TextPreprocessor(db=db)
    prep_stats = preprocessor.process_all_pending()
    logger.info(
        f"  Comentarios: {prep_stats['comments_valid']} válidos, "
        f"{prep_stats['comments_filtered']} filtrados"
    )
    logger.info(
        f"  Posts: {prep_stats['posts_valid']} válidos, "
        f"{prep_stats['posts_filtered']} filtrados"
    )

    # Paso 3: Mostrar ejemplos
    logger.info(f"\n[3/3] Ejemplos de texto procesado:")
    logger.info("-" * 40)

    sentiment_texts = preprocessor.get_texts_for_sentiment(limit=3)
    if sentiment_texts:
        logger.info("Sentimiento (RoBERTa):")
        for t in sentiment_texts[:3]:
            preview = t["text"][:120] + "..." if len(t["text"]) > 120 else t["text"]
            logger.info(f"  → {preview}")
    else:
        logger.warning("  No se encontraron textos en los últimos minutos")

    topic_texts = preprocessor.get_texts_for_topics(limit=3)
    if topic_texts:
        logger.info("\nTópicos (BERTopic):")
        for t in topic_texts[:3]:
            preview = t["text"][:120] + "..." if len(t["text"]) > 120 else t["text"]
            logger.info(f"  → {preview}")

    # Resumen final
    stats = db.get_stats()
    logger.info(f"\nBD total: {stats['total_posts']} posts, {stats['total_comments']} comentarios, "
                f"{stats['valid_preprocessed']} textos válidos")


def main():
    parser = argparse.ArgumentParser(description="Recolectar datos de Reddit")
    parser.add_argument(
        "--days", type=int, default=DEFAULT_COLLECTION_DAYS,
        help=f"Días hacia atrás para recolectar (default: {DEFAULT_COLLECTION_DAYS})"
    )
    parser.add_argument(
        "--subreddits", nargs="+", default=None,
        help=f"Subreddits objetivo (default: {TARGET_SUBREDDITS})"
    )
    parser.add_argument(
        "--max-posts", type=int, default=500,
        help="Máximo de posts por subreddit (default: 500)"
    )

    # Modo continuo
    parser.add_argument(
        "--continuous", action="store_true",
        help="Ejecutar en bucle continuo (Ctrl+C para detener)"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Segundos entre iteraciones en modo continuo (default: 30)"
    )

    # Modo live demo
    parser.add_argument(
        "--live", action="store_true",
        help="Modo demo: recolectar y preprocesar en una sola corrida"
    )
    parser.add_argument(
        "--minutes", type=int, default=5,
        help="Minutos hacia atrás para modo live (default: 5)"
    )

    # Modo histórico vía Arctic Shift API
    parser.add_argument(
        "--arctic", action="store_true",
        help="Recolección histórica usando Arctic Shift API (archivo público de Reddit, usar con --days)"
    )
    # Modo histórico vía PRAW timestamp search (limitado)
    parser.add_argument(
        "--historical", action="store_true",
        help="Recolección histórica vía búsqueda Lucene por timestamp (PRAW, sin límite por día)"
    )

    args = parser.parse_args()

    db = DatabaseManager()
    collector = RedditCollector(db=db)

    if args.arctic:
        from src.collection.arctic_collector import ArcticCollector
        arctic = ArcticCollector(db=db)
        subreddits = args.subreddits or TARGET_SUBREDDITS
        logger.info("=" * 60)
        logger.info("RECOLECCIÓN HISTÓRICA — Arctic Shift API")
        logger.info(f"Subreddits: {subreddits}")
        logger.info(f"Días: {args.days}")
        logger.info("=" * 60)
        for sub in subreddits:
            result = arctic.collect_historical(sub, days=args.days)
            dist = result.get("daily_distribution", {})
            days_ok = sum(1 for v in dist.values() if v > 0)
            logger.info(
                f"  r/{sub}: {result['new_posts_inserted']} posts, "
                f"{result['new_comments_inserted']} comentarios, "
                f"{days_ok}/{args.days} días con datos"
            )
        stats = db.get_stats()
        logger.info(f"BD final: {stats['total_posts']} posts, {stats['total_comments']} comentarios")

    elif args.historical:
        subreddits = args.subreddits or TARGET_SUBREDDITS
        logger.info("=" * 60)
        logger.info("RECOLECCIÓN HISTÓRICA — PRAW (sin límite por día)")
        logger.info(f"Subreddits: {subreddits}")
        logger.info(f"Días: {args.days}")
        logger.info("=" * 60)
        all_results = []
        for sub in subreddits:
            result = collector.collect_historical(
                sub,
                days=args.days,
                max_comments_per_post=args.max_posts,
            )
            all_results.append(result)
            dist = result.get("daily_distribution", {})
            days_ok = sum(1 for v in dist.values() if v > 0)
            logger.info(f"  r/{sub}: {result['new_posts_inserted']} posts, "
                        f"{result['new_comments_inserted']} comentarios, "
                        f"{days_ok}/{args.days} días con datos")
        stats = db.get_stats()
        logger.info(f"BD final: {stats['total_posts']} posts, {stats['total_comments']} comentarios")
    elif args.live:
        run_live_demo(collector, db, args)
    elif args.continuous:
        run_continuous(collector, db, args)
    else:
        run_once(collector, db, args)


if __name__ == "__main__":
    main()
