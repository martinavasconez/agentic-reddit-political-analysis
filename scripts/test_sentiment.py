"""
Script de prueba del Agente de Sentimiento.
Clasifica 100 comentarios e imprime resultados detallados.

Uso:
    python -m scripts.test_sentiment
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
from src.agents.sentiment.sentiment_agent import SentimentAgent
from src.database.db_manager import DatabaseManager


def main():
    db = DatabaseManager()
    agent = SentimentAgent(db=db)
    agent._load_models()

    # Tomar 100 textos ya analizados para mostrar resultados
    conn = sqlite3.connect("data/reddit_political.db")
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            pt.original_text,
            sr.roberta_label,
            sr.roberta_confidence,
            sr.decision,
            sr.final_label,
            sr.final_confidence,
            sr.vader_compound,
            sr.vader_label
        FROM sentiment_results sr
        JOIN preprocessed_texts pt
            ON sr.source_id = pt.source_id AND sr.source_type = pt.source_type
        ORDER BY RANDOM()
        LIMIT 100
    """).fetchall()
    conn.close()

    print(f"\n{'='*80}")
    print(f"{'CLASIFICACIÓN DE SENTIMIENTO — 100 COMENTARIOS ALEATORIOS':^80}")
    print(f"{'='*80}\n")

    label_icons = {
        "positive": "🟢",
        "negative": "🔴",
        "neutral":  "🟡",
        "ambiguous": "⚪",
    }

    decision_counts = {}
    label_counts = {}

    for i, r in enumerate(rows, 1):
        label = r["final_label"]
        decision = r["decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1

        icon = label_icons.get(label, "?")
        text_preview = r["original_text"].replace("\n", " ")

        vader_info = ""
        if r["vader_compound"] is not None:
            vader_info = f" | VADER={r['vader_compound']:+.3f} ({r['vader_label']})"

        print(f"[{i:03d}] {icon} {label.upper():<10} "
              f"conf={r['roberta_confidence']:.3f} "
              f"decision={r['decision']:<15}"
              f"{vader_info}")
        print(f"       {text_preview}")
        print()

    # Resumen
    total = len(rows)
    print(f"{'='*80}")
    print(f"{'RESUMEN':^80}")
    print(f"{'='*80}")
    print(f"\nDistribución de sentimiento:")
    for label in ["negative", "neutral", "positive", "ambiguous"]:
        count = label_counts.get(label, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label_icons[label]} {label:<10} {count:>4}  ({pct:5.1f}%)  {bar}")

    print(f"\nDecisiones ReAct:")
    for decision, count in sorted(decision_counts.items()):
        pct = count / total * 100
        print(f"  {decision:<20} {count:>4}  ({pct:5.1f}%)")

    print()


if __name__ == "__main__":
    main()
