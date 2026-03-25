"""
Inspección visual de detección de tendencias.

Muestra los tópicos detectados con sus textos asignados para evaluación manual.

Uso:
    python -m scripts.inspect_trends                  # top tópicos del último run
    python -m scripts.inspect_trends --topic 0        # ver textos del tópico 0
    python -m scripts.inspect_trends --n 5            # 5 textos por tópico
    python -m scripts.inspect_trends --decision emerging_trend
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DECISION_ICON = {
    "emerging_trend": "🔥 EMERGING TREND",
    "localized_spike": "⚡ LOCALIZED SPIKE",
    "moderate_trend": "📈 MODERATE TREND",
    "discarded": "·  DISCARDED",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=int, default=None, help="Ver textos de un tópico específico")
    parser.add_argument("--n", type=int, default=5, help="Textos por tópico (default: 5)")
    parser.add_argument("--decision", type=str, default=None,
                        help="Filtrar por decisión: emerging_trend, localized_spike, moderate_trend, discarded")
    args = parser.parse_args()

    conn = sqlite3.connect("data/reddit_political.db")
    conn.row_factory = sqlite3.Row

    # Obtener el último run
    run = conn.execute("""
        SELECT model_run_id, analyzed_at
        FROM trend_analysis
        ORDER BY analyzed_at DESC
        LIMIT 1
    """).fetchone()

    if not run:
        print("No hay resultados de tendencias en la BD. Corre primero: python -m scripts.run_trends")
        return

    run_id = run["model_run_id"]
    print(f"\n{'='*80}")
    print(f"  TENDENCIAS — run: {run_id[:8]}...  ({run['analyzed_at'][:19]})")
    print(f"{'='*80}\n")

    # Filtros
    where = f"WHERE ta.model_run_id = '{run_id}'"
    if args.topic is not None:
        where += f" AND ta.topic_id = {args.topic}"
    if args.decision:
        where += f" AND ta.trend_decision = '{args.decision}'"

    topics = conn.execute(f"""
        SELECT ta.topic_id, ta.topic_label, ta.trend_decision,
               ta.delta, ta.corpus_coverage, ta.n_current_texts
        FROM trend_analysis ta
        {where}
        ORDER BY ta.delta DESC
    """).fetchall()

    if not topics:
        print("No se encontraron tópicos con esos filtros.")
        return

    for t in topics:
        icon = DECISION_ICON.get(t["trend_decision"], "?")
        print(f"{icon}")
        print(f"  Tópico {t['topic_id']}: {t['topic_label']}")
        print(f"  Δ={t['delta']:+.2f}  cobertura={t['corpus_coverage']*100:.1f}%  n={t['n_current_texts']} textos")
        print()

        # Textos asignados a este tópico
        textos = conn.execute("""
            SELECT pt.original_text
            FROM topic_assignments tap
            JOIN preprocessed_texts pt
                ON tap.source_id = pt.source_id AND tap.source_type = pt.source_type
            WHERE tap.model_run_id = ? AND tap.topic_id = ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (run_id, t["topic_id"], args.n)).fetchall()

        for i, tx in enumerate(textos, 1):
            print(f"  [{i}] {tx['original_text']}")
            print()

        print("-" * 80)
        print()

    conn.close()


if __name__ == "__main__":
    main()
