"""
Script para ejecutar el Agente de Validación y Reporte.

Uso:
    python -m scripts.run_validation                        # Genera reporte con último run
    python -m scripts.run_validation --model-run-id abc123  # Usa run específico
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.agents.validation.validation_agent import ValidationAgent
from src.database.db_manager import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Agente de Validación y Reporte")
    parser.add_argument("--model-run-id", type=str, default=None,
                        help="ID del run de tendencias a evaluar (default: último)")
    args = parser.parse_args()

    db = DatabaseManager()
    agent = ValidationAgent(db=db)
    result = agent.run(model_run_id=args.model_run_id)

    if result["status"] == "completed":
        logger.info(f"Reporte generado en: {result['report_path']}")
        print(f"\nReporte disponible en: {result['report_path']}")
        print("\nHallazgos:")
        for f in result.get("findings", []):
            print(f"  • {f}")
    else:
        logger.warning("No se generó reporte (sin datos).")


if __name__ == "__main__":
    main()
