"""
Etiquetado de ground truth para sentimiento usando DeepSeek V3.

Uso:
    python -m scripts.label_ground_truth                        # prueba con 10 textos (no guarda)
    python -m scripts.label_ground_truth --n 50 --save          # guarda lote de 50
    python -m scripts.label_ground_truth --all --save           # etiqueta todo el corpus
    python -m scripts.label_ground_truth --all --save --workers 20  # con 20 threads (más rápido)
    python -m scripts.label_ground_truth --stats                # distribución de lo ya etiquetado
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from openai import OpenAI

from config.settings import DEEPSEEK_API_KEY, DB_PATH

LLM_MODEL = "deepseek-chat"  # DeepSeek-V3
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = """
You are an expert annotator for political sentiment analysis research.
Your task is to classify the sentiment of Reddit comments from r/politics.
LABELS:
- positive: expresses approval, praise, support, optimism, or satisfaction toward a political actor, policy, party, or outcome.
- negative: expresses criticism, mockery, anger, distrust, disapproval, or opposition toward a political actor, policy, party, or outcome.
- neutral: factual, explanatory, speculative, or conversational comments without clear approval or criticism.
IMPORTANT INTERPRETATION RULES:
1. Determine the TARGET of the comment (person, party, policy, institution).
2. Determine whether the author SUPPORTS or CRITICIZES that target.
3. If the author criticizes or mocks a political actor → negative.

SARCASM AND MOCKERY RULES:
4. If the comment contains praise that is clearly ironic or mocking, classify by the INTENDED meaning.
   Example: "Great job ruining everything again" → negative.
5. Statements that imagine a political opponent failing, crying, or being humiliated are MOCKERY → negative.
6. Polite phrases used to dismiss someone in an argument (e.g., "sure you're completely right, have a good day") are DISMISSIVE → neutral.

CRITICISM RULES:
7. Questioning the truth of political claims, statistics, or narratives counts as criticism → negative.
8. Saying a party/candidate should replace their candidate → negative.
9. Speculation about events without approval or criticism → neutral.

FINAL DECISION RULE:
10. If approval or praise is genuine → positive.
11. If criticism, distrust, ridicule, or disapproval appears → negative.
12. If the comment is purely informational, speculative, or dismissive without stance → neutral.
13. If the comment mainly clarifies, corrects, or debates facts without clear approval or criticism toward a political actor, classify as neutral.
14. Direct insults toward political actors → negative.
15. General discussion of events, history, or political processes without clear judgment → neutral.
Respond ONLY with valid JSON:
{"label": "positive" | "negative" | "neutral", "reasoning": "one short sentence"}
"""

def _call_deepseek(client: OpenAI, text: str) -> dict:
    """Llama a DeepSeek y retorna {'label': ..., 'reasoning': ...}."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify the sentiment of this Reddit comment:\n\n{text}"},
        ],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def _fetch_texts(conn: sqlite3.Connection, limit: int, unlabeled_only: bool) -> list[dict]:
    if unlabeled_only:
        rows = conn.execute("""
            SELECT pt.source_id, pt.source_type, pt.original_text, pt.subreddit
            FROM preprocessed_texts pt
            LEFT JOIN ground_truth_labels gt
                ON pt.source_id = gt.source_id AND pt.source_type = gt.source_type
            WHERE pt.is_valid = 1 AND gt.id IS NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT source_id, source_type, original_text, subreddit
            FROM preprocessed_texts
            WHERE is_valid = 1
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def _ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ground_truth_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            original_text TEXT NOT NULL,
            llm_label TEXT NOT NULL,
            llm_reasoning TEXT,
            model_used TEXT NOT NULL,
            labeled_at TEXT NOT NULL,
            UNIQUE(source_id, source_type)
        )
    """)
    conn.commit()


def _save_label(conn: sqlite3.Connection, lock: Lock, source_id: str, source_type: str,
                original_text: str, label: str, reasoning: str):
    with lock:
        conn.execute("""
            INSERT OR IGNORE INTO ground_truth_labels
                (source_id, source_type, original_text, llm_label, llm_reasoning, model_used, labeled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source_id, source_type, original_text, label, reasoning, LLM_MODEL,
              datetime.utcnow().isoformat()))
        conn.commit()


def _process_one(item: tuple) -> dict | None:
    """Worker: llama a DeepSeek para un texto. Retorna resultado o None si falla."""
    t, client = item
    text = t["original_text"]
    if len(text) > 1500:
        text = text[:1500] + "..."
    try:
        result = _call_deepseek(client, text)
        label = result.get("label", "").lower()
        reasoning = result.get("reasoning", "")
        if label not in ("positive", "negative", "neutral"):
            return None
        return {**t, "label": label, "reasoning": reasoning}
    except Exception as e:
        logger.warning(f"Error en {t['source_id']}: {e}")
        return None


def show_stats(conn: sqlite3.Connection):
    total = conn.execute("SELECT COUNT(*) FROM ground_truth_labels").fetchone()[0]
    if total == 0:
        print("No hay ground truth labels aún.")
        return

    print(f"\n{'='*60}")
    print(f"  GROUND TRUTH — {total:,} textos etiquetados")
    print(f"{'='*60}\n")

    rows = conn.execute("""
        SELECT llm_label, COUNT(*) as cnt
        FROM ground_truth_labels
        GROUP BY llm_label ORDER BY cnt DESC
    """).fetchall()

    icons = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
    for r in rows:
        pct = r["cnt"] / total * 100
        bar = "█" * int(pct / 2)
        icon = icons.get(r["llm_label"], "?")
        print(f"  {icon} {r['llm_label']:<12} {r['cnt']:>6} ({pct:5.1f}%)  {bar}")
    print()


def _print_result(i: int, text: str, label: str, reasoning: str):
    icons = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
    icon = icons.get(label, "?")
    preview = text[:200].replace("\n", " ")
    print(f"\n[{i}] {icon} {label.upper()}")
    print(f"     Texto  : {preview}{'...' if len(text) > 200 else ''}")
    print(f"     Razón  : {reasoning}")
    print(f"     {'-'*55}")


def run_labeling(n: int, save: bool, all_texts: bool, workers: int):
    if not DEEPSEEK_API_KEY:
        logger.error("Falta DEEPSEEK_API_KEY en .env")
        return

    # Cada thread necesita su propio cliente OpenAI
    clients = [OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
               for _ in range(workers)]

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_table(conn)
    db_lock = Lock()

    limit = 999999 if all_texts else n
    texts = _fetch_texts(conn, limit=limit, unlabeled_only=save)

    if not texts:
        logger.info("No hay textos pendientes de etiquetar.")
        conn.close()
        return

    mode = "guardar en BD" if save else "solo imprimir (modo prueba)"
    logger.info(f"Textos: {len(texts):,} | workers: {workers} | modo: {mode}")
    if not save:
        logger.info("Agrega --save para guardar los resultados.")

    # Asignar cliente round-robin a cada texto
    items = [(t, clients[i % workers]) for i, t in enumerate(texts)]

    results = []
    errors = 0
    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, item): item[0] for item in items}

        for future in as_completed(futures):
            completed += 1
            result = future.result()

            if result is None:
                errors += 1
            else:
                results.append(result)
                if save:
                    _save_label(conn, db_lock,
                                result["source_id"], result["source_type"],
                                result["original_text"], result["label"], result["reasoning"])
                if not save or completed <= 10:
                    _print_result(completed, result["original_text"],
                                  result["label"], result["reasoning"])

            if completed % 500 == 0:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                remaining = (len(texts) - completed) / rate
                logger.info(
                    f"  Progreso: {completed:,}/{len(texts):,} "
                    f"({rate:.1f} req/s) — "
                    f"faltan ~{remaining/60:.0f} min"
                )

    conn.close()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Etiquetados : {len(results):,}  |  Errores: {errors}")
    print(f"  Tiempo total: {elapsed/60:.1f} min  ({elapsed/3600:.1f} h)")
    if save:
        print(f"  Guardados en ground_truth_labels ✅")
    else:
        print(f"  Modo prueba — no se guardó nada.")
        print(f"  Si se ven bien, corre con --save para guardar.")
    print(f"{'='*60}\n")

    if results:
        dist = Counter(r["label"] for r in results)
        icons = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
        print("  Distribución del lote:")
        for label, cnt in dist.most_common():
            pct = cnt / len(results) * 100
            print(f"  {icons.get(label,'?')} {label:<12} {cnt:>6} ({pct:.1f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Etiquetado de ground truth con DeepSeek V3")
    parser.add_argument("--n", type=int, default=10,
                        help="Número de textos a etiquetar (default: 10)")
    parser.add_argument("--save", action="store_true",
                        help="Guardar resultados en BD (default: solo imprime)")
    parser.add_argument("--all", action="store_true",
                        help="Etiquetar todos los textos pendientes")
    parser.add_argument("--workers", type=int, default=1,
                        help="Threads concurrentes (default: 1, recomendado: 20 para corpus completo)")
    parser.add_argument("--stats", action="store_true",
                        help="Ver distribución de labels ya guardados")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if args.stats:
        _ensure_table(conn)
        show_stats(conn)
        conn.close()
        return

    conn.close()
    run_labeling(n=args.n, save=args.save, all_texts=args.all, workers=args.workers)


if __name__ == "__main__":
    main()
