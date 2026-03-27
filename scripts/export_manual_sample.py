"""
Exporta muestra aleatoria de 300 textos con label DeepSeek para validación manual.
Genera un CSV con columna vacía 'manual_label' para que el evaluador clasifique.

Uso:
    python -m scripts.export_manual_sample
    python -m scripts.export_manual_sample --size 300 --seed 42 --output sample_manual.csv
"""

import argparse
import csv
import random
import sqlite3
from pathlib import Path

DB_PATH = "data/reddit_political.db"
DEFAULT_OUTPUT = "data/evaluation/manual_validation_sample.csv"
DEFAULT_SIZE = 300
DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Exportar muestra para validación manual")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Número de textos a exportar")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla aleatoria (para reproducibilidad)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Ruta del CSV de salida")
    args = parser.parse_args()

    random.seed(args.seed)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Extraer todos los IDs disponibles que tienen label DeepSeek
    rows = conn.execute("""
        SELECT
            gt.id          AS row_id,
            gt.source_id,
            gt.source_type,
            gt.original_text,
            gt.llm_label   AS deepseek_label,
            gt.llm_reasoning AS deepseek_reasoning,
            sr.final_label AS roberta_label,
            sr.roberta_confidence
        FROM ground_truth_labels gt
        LEFT JOIN sentiment_results sr
            ON gt.source_id = sr.source_id AND gt.source_type = sr.source_type
        WHERE gt.llm_label IN ('negative', 'neutral', 'positive')
          AND LENGTH(gt.original_text) > 20
        ORDER BY RANDOM()
        LIMIT ?
    """, (args.size,)).fetchall()

    conn.close()

    if not rows:
        print("ERROR: No se encontraron filas en ground_truth_labels.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "row_id",
        "source_id",
        "source_type",
        "original_text",
        "deepseek_label",
        "deepseek_reasoning",
        "roberta_label",
        "roberta_confidence",
        "manual_label",       # <-- columna vacía para llenar
        "notas",              # <-- columna vacía para comentarios opcionales
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "row_id": row["row_id"],
                "source_id": row["source_id"],
                "source_type": row["source_type"],
                "original_text": row["original_text"],
                "deepseek_label": row["deepseek_label"],
                "deepseek_reasoning": row["deepseek_reasoning"],
                "roberta_label": row["roberta_label"] or "",
                "roberta_confidence": round(row["roberta_confidence"], 4) if row["roberta_confidence"] else "",
                "manual_label": "",
                "notas": "",
            })

    print(f"✅ Muestra exportada: {output_path}")
    print(f"   Textos: {len(rows)}")
    print(f"   Semilla: {args.seed}")
    print()
    print("Instrucciones:")
    print("  - Abre el CSV en Excel o Google Sheets")
    print("  - Llena la columna 'manual_label' con: negative / neutral / positive")
    print("  - La columna 'notas' es opcional para casos dudosos")
    print()
    print("Distribución DeepSeek en la muestra:")
    from collections import Counter
    dist = Counter(row["deepseek_label"] for row in rows)
    for label, count in sorted(dist.items()):
        print(f"  {label:10s}: {count:4d} ({count/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
