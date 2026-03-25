"""
Inspección visual del ground truth vs clasificación de RoBERTa.

Uso:
    python -m scripts.inspect_ground_truth          # todos los etiquetados
    python -m scripts.inspect_ground_truth --n 20   # últimos 20
    python -m scripts.inspect_ground_truth --wrong  # solo los que no coinciden
    python -m scripts.inspect_ground_truth --label negative  # filtrar por label DeepSeek
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_PATH

ICONS = {"positive": "🟢", "negative": "🔴", "neutral": "🟡", "ambiguous": "⚪"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Número de textos a mostrar")
    parser.add_argument("--wrong", action="store_true", help="Solo mostrar los incorrectos")
    parser.add_argument("--label", type=str, default=None,
                        help="Filtrar por label DeepSeek: positive, negative, neutral")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            gt.source_id,
            gt.llm_label        AS ground_truth,
            gt.llm_reasoning    AS gt_reasoning,
            sr.final_label      AS roberta_label,
            sr.roberta_confidence,
            sr.decision,
            gt.original_text
        FROM ground_truth_labels gt
        JOIN sentiment_results sr
            ON gt.source_id = sr.source_id
            AND gt.source_type = sr.source_type
        ORDER BY gt.labeled_at DESC
    """).fetchall()
    conn.close()

    if not rows:
        print("No hay ground truth guardado aún. Corre: python -m scripts.label_ground_truth --save")
        return

    # Filtros
    if args.wrong:
        rows = [r for r in rows if r["ground_truth"] != r["roberta_label"]]
    if args.label:
        rows = [r for r in rows if r["ground_truth"] == args.label]
    if args.n:
        rows = rows[:args.n]

    total = len(rows)
    correct = sum(1 for r in rows if r["ground_truth"] == r["roberta_label"]
                  and r["roberta_label"] != "ambiguous")
    ambiguous = sum(1 for r in rows if r["roberta_label"] == "ambiguous")
    comparable = total - ambiguous

    print(f"\n{'='*80}")
    print(f"  GROUND TRUTH vs RoBERTa — {total} textos")
    if comparable > 0:
        print(f"  Accuracy: {correct}/{comparable} = {correct/comparable*100:.1f}%  "
              f"(excluyendo {ambiguous} ambiguous)")
    print(f"{'='*80}\n")

    for i, r in enumerate(rows, 1):
        match = "✅" if r["ground_truth"] == r["roberta_label"] else "❌"
        gt_icon = ICONS.get(r["ground_truth"], "?")
        rb_icon = ICONS.get(r["roberta_label"], "?")

        print(f"[{i:>3}] {match}  "
              f"DEEPSEEK: {gt_icon} {r['ground_truth']:<10}  "
              f"ROBERTA: {rb_icon} {r['roberta_label']:<10}  "
              f"conf={r['roberta_confidence']:.2f}  [{r['decision']}]  "
              f"id={r['source_id']}")
        print(f"       Texto  : {r['original_text'].replace(chr(10), ' ')}")
        print(f"       Razón  : {r['gt_reasoning']}")
        print()

    # Resumen de errores por tipo
    print(f"{'='*80}")
    print("  ERRORES POR TIPO:")
    from collections import Counter
    errors = [(r["ground_truth"], r["roberta_label"])
              for r in rows if r["ground_truth"] != r["roberta_label"]]
    for (gt, rb), cnt in Counter(errors).most_common():
        print(f"  {ICONS.get(gt,'?')}{gt} → {ICONS.get(rb,'?')}{rb} : {cnt} caso(s)")
    print()


if __name__ == "__main__":
    main()
