"""
Script para ejecutar el Orquestador LangGraph (sistema agentic completo).

Uso:
    python -m scripts.run_orchestrator                                  # Flujo completo
    python -m scripts.run_orchestrator --limit-sentiment 5000           # Limitar sentimiento
    python -m scripts.run_orchestrator --limit-trends 200000            # Limitar tendencias
    python -m scripts.run_orchestrator --skip-preprocess                # Saltar preprocesamiento
    python -m scripts.run_orchestrator --results                        # Ver último run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.database.db_manager import DatabaseManager


def show_results():
    """Muestra resultados del último run de orquestación."""
    db = DatabaseManager()
    conn = db._get_connection()
    try:
        row = conn.execute("""
            SELECT * FROM orchestration_runs
            ORDER BY started_at DESC LIMIT 1
        """).fetchone()
        if not row:
            print("No hay runs de orquestación registrados.")
            return
        row = dict(row)
        print(f"\nÚltimo run: {row['run_id']}")
        print(f"  Estado:  {row['status']}")
        print(f"  Inicio:  {row['started_at']}")
        print(f"  Fin:     {row.get('finished_at', 'N/A')}")
        if row.get("steps_completed"):
            steps = json.loads(row["steps_completed"])
            print(f"  Pasos:   {' → '.join(steps)}")
        if row.get("results_json"):
            results = json.loads(row["results_json"])
            print(f"  Resultados:")
            for key, val in results.items():
                if isinstance(val, dict):
                    print(f"    {key}:")
                    for k, v in val.items():
                        if not isinstance(v, (dict, list)):
                            print(f"      {k}: {v}")
                else:
                    print(f"    {key}: {val}")
        if row.get("error_message"):
            print(f"  Error: {row['error_message']}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Orquestador LangGraph (sistema agentic)")
    parser.add_argument("--limit-sentiment", type=int, default=1000,
                        help="Máximo de textos para sentimiento (default: 1000)")
    parser.add_argument("--limit-trends", type=int, default=50000,
                        help="Máximo de textos para tendencias (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamaño de lote para RoBERTa (default: 64)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Saltar preprocesamiento (usar datos existentes)")
    parser.add_argument("--results", action="store_true",
                        help="Mostrar resultados del último run")
    args = parser.parse_args()

    if args.results:
        show_results()
        return

    from src.orchestrator.orchestrator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator()
    final_state = orchestrator.run(
        limit_sentiment=args.limit_sentiment,
        limit_trends=args.limit_trends,
        batch_size=args.batch_size,
        skip_preprocess=args.skip_preprocess,
    )

    print(f"\nRun completado: {final_state.get('run_id')}")
    print(f"Pasos: {' → '.join(final_state.get('steps_completed', []))}")
    if final_state.get("validation_result", {}).get("report_path"):
        print(f"Reporte: {final_state['validation_result']['report_path']}")


if __name__ == "__main__":
    main()
