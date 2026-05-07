"""
Muestra ejemplos comparativos del preprocesamiento:
  TEXTO ORIGINAL → text_for_sentiment (RoBERTa) → text_for_topics (BERTopic)

Uso:
    python -m scripts.inspect_preprocessing
    python -m scripts.inspect_preprocessing --n 5
    python -m scripts.inspect_preprocessing --with-url    # solo textos con URLs
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Inspeccionar preprocesamiento")
    parser.add_argument("--n", type=int, default=3, help="Número de ejemplos (default: 3)")
    parser.add_argument("--with-url", action="store_true", help="Solo textos que contenían URLs")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if args.with_url:
        # Textos donde sentiment tiene 'http' (URL reemplazada) pero topics no
        query = """
            SELECT original_text, text_for_sentiment, text_for_topics, word_count
            FROM preprocessed_texts
            WHERE is_valid = 1
              AND text_for_sentiment LIKE '%http%'
              AND text_for_topics NOT LIKE '%http%'
              AND original_text NOT LIKE '%I am a bot%'
              AND original_text NOT LIKE '%Thank you for participating%'
              AND word_count BETWEEN 15 AND 80
            ORDER BY RANDOM()
            LIMIT ?
        """
    else:
        query = """
            SELECT original_text, text_for_sentiment, text_for_topics, word_count
            FROM preprocessed_texts
            WHERE is_valid = 1
              AND original_text NOT LIKE '%I am a bot%'
              AND original_text NOT LIKE '%Thank you for participating%'
              AND word_count BETWEEN 15 AND 80
            ORDER BY RANDOM()
            LIMIT ?
        """

    rows = conn.execute(query, (args.n,)).fetchall()
    conn.close()

    for i, r in enumerate(rows, 1):
        sep = "-" * 80
        print(f"\n  TEXTO ORIGINAL  ({r['word_count']} palabras)")
        print(f"  {sep}")
        print(f"  {r['original_text'][:500]}")
        print()
        print(f"    text_for_sentiment  (RoBERTa)")
        print(f"    {sep}")
        print(f"    {r['text_for_sentiment'][:500]}")
        print()
        print(f"    text_for_topics     (BERTopic)")
        print(f"    {sep}")
        print(f"    {r['text_for_topics'][:500]}")
        print()
        if i < len(rows):
            print("=" * 90)


if __name__ == "__main__":
    main()
