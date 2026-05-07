"""
Script para ejecutar el Pipeline Tradicional (sin comportamiento agentic).

Uso:
    python -m scripts.run_pipeline                          # Pipeline completo
    python -m scripts.run_pipeline --limit-sentiment 5000   # Limitar sentimiento
    python -m scripts.run_pipeline --limit-trends 50000     # Limitar tendencias
    python -m scripts.run_pipeline --sentiment-only         # Solo sentimiento
    python -m scripts.run_pipeline --trends-only            # Solo tendencias
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.pipeline.traditional_pipeline import TraditionalPipeline
from src.database.db_manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Pipeline Tradicional (sin agentic)")
    parser.add_argument("--limit-sentiment", type=int, default=1000,
                        help="Máximo de textos para sentimiento (default: 1000)")
    parser.add_argument("--limit-trends", type=int, default=50000,
                        help="Máximo de textos para tendencias (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño de lote para RoBERTa (default: 64)")
    parser.add_argument("--current-days", type=float, default=2,
                        help="Días de ventana actual para tendencias (default: 2)")
    parser.add_argument("--sentiment-only", action="store_true",
                        help="Solo ejecutar sentimiento")
    parser.add_argument("--trends-only", action="store_true",
                        help="Solo ejecutar tendencias")
    args = parser.parse_args()

    db = DatabaseManager()
    pipeline = TraditionalPipeline(db=db)

    if args.sentiment_only:
        result = pipeline.run_sentiment(
            limit=args.limit_sentiment,
            batch_size=args.batch_size,
        )
        logger.info(f"Resultado sentimiento: {result}")
    elif args.trends_only:
        result = pipeline.run_trends(
            limit=args.limit_trends,
            current_days=args.current_days,
        )
        logger.info(f"Resultado tendencias: {result.get('n_topics', 0)} tópicos")
    else:
        result = pipeline.run_full(
            limit_sentiment=args.limit_sentiment,
            limit_trends=args.limit_trends,
            current_days=args.current_days,
            batch_size=args.batch_size,
        )
        logger.info(f"Pipeline completo finalizado. Reporte: {result.get('report_path')}")


if __name__ == "__main__":
    main()
