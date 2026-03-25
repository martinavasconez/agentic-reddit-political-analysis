"""
Script para mostrar ejemplos reales de preprocesamiento del corpus.
Busca comentarios donde la diferencia entre text_for_sentiment y text_for_topics
sea visible (URLs, menciones, o contenido político representativo).
"""

import sqlite3
from config.settings import DB_PATH

def get_examples():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 1. Comentarios con diferencias visibles (tenían URLs o menciones)
    with_diff = conn.execute("""
        SELECT original_text, text_for_sentiment, text_for_topics, word_count
        FROM preprocessed_texts
        WHERE source_type = 'comment'
          AND is_valid = 1
          AND text_for_sentiment != text_for_topics
          AND word_count BETWEEN 20 AND 60
        ORDER BY RANDOM()
        LIMIT 5
    """).fetchall()

    # 2. Comentarios políticos representativos (sin diferencias pero buen contenido)
    general = conn.execute("""
        SELECT original_text, text_for_sentiment, text_for_topics, word_count
        FROM preprocessed_texts
        WHERE source_type = 'comment'
          AND is_valid = 1
          AND word_count BETWEEN 25 AND 55
        ORDER BY RANDOM()
        LIMIT 5
    """).fetchall()

    conn.close()
    return with_diff, general


def print_example(row, idx, label=""):
    sep = "-" * 80
    print(f"\n{'=' * 80}")
    print(f"  EJEMPLO {idx}{f'  [{label}]' if label else ''}")
    print(f"{'=' * 80}")
    print(f"\n  TEXTO ORIGINAL  ({row['word_count']} palabras)")
    print(f"  {sep}")
    print(f"  {row['original_text'][:300]}{'...' if len(row['original_text']) > 300 else ''}")

    print(f"\n  text_for_sentiment  (RoBERTa)")
    print(f"  {sep}")
    print(f"  {row['text_for_sentiment'][:300]}{'...' if len(row['text_for_sentiment']) > 300 else ''}")

    print(f"\n  text_for_topics     (BERTopic)")
    print(f"  {sep}")
    print(f"  {row['text_for_topics'][:300]}{'...' if len(row['text_for_topics']) > 300 else ''}")

    if row['text_for_sentiment'] != row['text_for_topics']:
        print(f"\n  >> Las dos versiones DIFIEREN (habia URLs o @menciones en el original)")
    else:
        print(f"\n  >> Las dos versiones son iguales (no habia URLs ni menciones)")
    print()


def main():
    with_diff, general = get_examples()

    print("\n" + "█" * 80)
    print("  EJEMPLOS DE PREPROCESAMIENTO — CORPUS r/politics")
    print("█" * 80)

    if with_diff:
        print("\n\n▶  SECCIÓN A: Comentarios con diferencia entre rutas (tenían URLs o @menciones)\n")
        for i, row in enumerate(with_diff, 1):
            print_example(row, i, "con diferencia")
    else:
        print("\n  (No se encontraron comentarios con URLs/menciones en este muestreo)\n")

    print("\n\n▶  SECCIÓN B: Comentarios políticos representativos\n")
    for i, row in enumerate(general, 1):
        print_example(row, i, "general")


if __name__ == "__main__":
    main()
