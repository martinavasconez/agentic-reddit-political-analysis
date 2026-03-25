"""
Script para ejecutar el Agente de Análisis de Sentimiento.

Uso:
    python -m scripts.run_sentiment                      # Analiza hasta 1000 textos
    python -m scripts.run_sentiment --limit 500          # Límite personalizado
    python -m scripts.run_sentiment --high-conf 0.9      # Ajustar umbral alto
    python -m scripts.run_sentiment --low-conf 0.5       # Ajustar umbral bajo
    python -m scripts.run_sentiment --stats              # Solo muestra estadísticas
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.agents.sentiment.sentiment_agent import SentimentAgent, HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD
from src.database.db_manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Agente de Análisis de Sentimiento")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Máximo de textos a analizar (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño de lote para RoBERTa (default: 64)")
    parser.add_argument("--high-conf", type=float, default=HIGH_CONF_THRESHOLD,
                        help=f"Umbral confianza alta (default: {HIGH_CONF_THRESHOLD})")
    parser.add_argument("--low-conf", type=float, default=LOW_CONF_THRESHOLD,
                        help=f"Umbral confianza baja (default: {LOW_CONF_THRESHOLD})")
    parser.add_argument("--stats", action="store_true",
                        help="Solo mostrar estadísticas de la BD sin analizar")
    args = parser.parse_args()

    db = DatabaseManager()

    if args.stats:
        stats = db.get_sentiment_stats()
        logger.info("=" * 50)
        logger.info("ESTADÍSTICAS DE SENTIMIENTO EN BD")
        logger.info(f"  Total analizados   : {stats['total_analyzed']}")
        logger.info(f"  Distribución labels: {stats['label_distribution']}")
        logger.info(f"  Distribución decis.: {stats['decision_distribution']}")
        logger.info(f"  Confianza promedio : {stats['avg_confidence']}")
        logger.info(f"  % Ambiguos         : {stats['pct_ambiguous']}%")
        logger.info("=" * 50)
        return

    agent = SentimentAgent(
        db=db,
        high_conf=args.high_conf,
        low_conf=args.low_conf,
    )

    logger.info("=" * 60)
    logger.info("AGENTE DE ANÁLISIS DE SENTIMIENTO")
    logger.info(f"  Umbrales: high={args.high_conf} | low={args.low_conf}")
    logger.info(f"  Límite: {args.limit} textos")
    logger.info("=" * 60)

    agent.run(limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
