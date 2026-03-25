"""
Inspección visual de clasificación de sentimiento.

Lee los resultados ya clasificados de la BD y muestra texto + label.

Uso:
    python -m scripts.inspect_sentiment           # 20 comentarios
    python -m scripts.inspect_sentiment --n 50    # 50 comentarios
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LABEL_ICON = {
    "positive": "🟢 POSITIVE",
    "negative": "🔴 NEGATIVE",
    "neutral":  "🟡 NEUTRAL ",
    "ambiguous": "⚪ AMBIGUOUS",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Cantidad de textos (default: 20)")
    args = parser.parse_args()

    conn = sqlite3.connect("data/reddit_political.db")
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            pt.original_text,
            pt.subreddit,
            pt.source_type,
            sr.final_label,
            sr.roberta_confidence,
            sr.decision,
            sr.vader_compound,
            sr.vader_label
        FROM sentiment_results sr
        JOIN preprocessed_texts pt
            ON sr.source_id = pt.source_id AND sr.source_type = pt.source_type
        ORDER BY RANDOM()
        LIMIT ?
    """, (args.n,)).fetchall()
    conn.close()

    print(f"\n{'='*80}")
    print(f"  SENTIMIENTO — {len(rows)} textos")
    print(f"{'='*80}\n")

    for i, r in enumerate(rows, 1):
        icon = LABEL_ICON.get(r["final_label"], "?")
        vader_info = ""
        if r["vader_compound"] is not None:
            vader_info = f" | VADER={r['vader_compound']:+.3f} ({r['vader_label']})"

        print(f"[{i:03d}] {icon}  conf={r['roberta_confidence']:.3f}  {r['decision']}{vader_info}")
        print(f"      {r['original_text']}")
        print()


if __name__ == "__main__":
    main()
