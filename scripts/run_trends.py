"""
Script para ejecutar el Agente de Detección de Tendencias.

Uso:
    python -m scripts.run_trends                       # Corre con defaults
    python -m scripts.run_trends --limit 20000         # Limitar textos
    python -m scripts.run_trends --n-topics 30         # Forzar N tópicos
    python -m scripts.run_trends --current-days 7      # Ventana actual en días
    python -m scripts.run_trends --results             # Ver resultados del último run
    python -m scripts.run_trends --coherence           # Calcular coherencia c_v y UMass
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.agents.trends.trends_agent import (
    TrendsAgent,
    DELTA_HIGH,
    DELTA_MODERATE,
    COVERAGE_THRESHOLD,
    HISTORICAL_DAYS,
    CURRENT_DAYS,
)
from src.database.db_manager import DatabaseManager


def show_results(db: DatabaseManager, run_id: str):
    results = db.get_trend_results(run_id)
    if not results:
        logger.warning(f"No hay resultados para run_id={run_id}")
        return

    print(f"\n{'='*70}")
    print(f"{'RESULTADOS DE TENDENCIAS — run_id: ' + run_id:^70}")
    print(f"{'='*70}\n")

    decision_icons = {
        "emerging_trend": "🔥",
        "localized_spike": "⚡",
        "moderate_trend": "📈",
        "discarded": "·",
    }

    by_decision = {}
    for r in results:
        by_decision.setdefault(r["trend_decision"], []).append(r)

    for decision in ["emerging_trend", "localized_spike", "moderate_trend", "discarded"]:
        items = by_decision.get(decision, [])
        if not items:
            continue
        icon = decision_icons[decision]
        print(f"{icon} {decision.upper().replace('_', ' ')} ({len(items)} tópicos):")
        for r in sorted(items, key=lambda x: x["delta"], reverse=True):
            label = r["topic_label"] or f"topic_{r['topic_id']}"
            print(
                f"   Δ={r['delta']:+6.2f}  cov={r['corpus_coverage']:.1%}  "
                f"curr={r['current_weight']:.4f}  hist_mean={r['historical_mean']:.4f}  "
                f"n={r['n_current_texts']:>4}  {label}"
            )
        print()


def calculate_coherence(run_id: str):
    """Calcula coherencia c_v y UMass para el modelo del run_id dado."""
    import sqlite3
    import json

    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora.dictionary import Dictionary
    except ImportError:
        logger.error("gensim no instalado. Ejecuta: pip install gensim")
        return

    conn = sqlite3.connect("data/reddit_political.db")
    conn.row_factory = sqlite3.Row

    # Obtener textos del run
    rows = conn.execute("""
        SELECT pt.text_for_topics, ta.topic_id
        FROM topic_assignments ta
        JOIN preprocessed_texts pt
            ON ta.source_id = pt.source_id AND ta.source_type = pt.source_type
        WHERE ta.model_run_id = ? AND ta.topic_id != -1
    """, (run_id,)).fetchall()

    # Obtener tópicos (palabras clave)
    trend_rows = conn.execute("""
        SELECT topic_id, topic_label
        FROM trend_analysis
        WHERE model_run_id = ?
        ORDER BY delta DESC
    """, (run_id,)).fetchall()
    conn.close()

    if not rows:
        logger.warning("Sin datos para calcular coherencia.")
        return

    texts_tokenized = [r["text_for_topics"].lower().split() for r in rows]
    dictionary = Dictionary(texts_tokenized)
    corpus = [dictionary.doc2bow(t) for t in texts_tokenized]

    # Extraer palabras clave por tópico del label (formato: "0_word1_word2_word3")
    topic_words = []
    for r in trend_rows:
        label = r["topic_label"] or ""
        words = [w for w in label.split("_") if w and not w.isdigit()]
        if words:
            topic_words.append(words)

    if len(topic_words) < 2:
        logger.warning("Pocos tópicos para calcular coherencia.")
        return

    # c_v
    try:
        cm_cv = CoherenceModel(
            topics=topic_words,
            texts=texts_tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        cv_score = cm_cv.get_coherence()
        logger.info(f"Coherencia c_v   : {cv_score:.4f}  (ideal > 0.55)")
    except Exception as e:
        logger.warning(f"No se pudo calcular c_v: {e}")
        cv_score = None

    # UMass
    try:
        cm_umass = CoherenceModel(
            topics=topic_words,
            corpus=corpus,
            dictionary=dictionary,
            coherence="u_mass",
        )
        umass_score = cm_umass.get_coherence()
        logger.info(f"Coherencia UMass : {umass_score:.4f}  (ideal > -2.0)")
    except Exception as e:
        logger.warning(f"No se pudo calcular UMass: {e}")
        umass_score = None

    return {"c_v": cv_score, "u_mass": umass_score}


def main():
    parser = argparse.ArgumentParser(description="Agente de Detección de Tendencias")
    parser.add_argument("--limit", type=int, default=50000,
                        help="Máximo de textos a cargar (default: 50000)")
    parser.add_argument("--n-topics", type=int, default=None,
                        help="Forzar número de tópicos (default: auto)")
    parser.add_argument("--historical-days", type=int, default=HISTORICAL_DAYS,
                        help=f"Días de ventana histórica/entrenamiento (default: {HISTORICAL_DAYS})")
    parser.add_argument("--current-days", type=int, default=CURRENT_DAYS,
                        help=f"Días de ventana de evaluación (default: {CURRENT_DAYS})")
    parser.add_argument("--delta-high", type=float, default=DELTA_HIGH,
                        help=f"Umbral Δ alto (default: {DELTA_HIGH})")
    parser.add_argument("--delta-moderate", type=float, default=DELTA_MODERATE,
                        help=f"Umbral Δ moderado (default: {DELTA_MODERATE})")
    parser.add_argument("--coverage", type=float, default=COVERAGE_THRESHOLD,
                        help=f"Umbral cobertura (default: {COVERAGE_THRESHOLD})")
    parser.add_argument("--results", action="store_true",
                        help="Mostrar resultados del último run sin reejecutar")
    parser.add_argument("--coherence", action="store_true",
                        help="Calcular métricas de coherencia (requiere gensim)")
    args = parser.parse_args()

    db = DatabaseManager()

    if args.results or args.coherence:
        run_id = db.get_latest_topic_model_run()
        if not run_id:
            logger.error("No hay runs previos en la BD.")
            return
        if args.results:
            show_results(db, run_id)
        if args.coherence:
            calculate_coherence(run_id)
        return

    agent = TrendsAgent(
        db=db,
        n_topics=args.n_topics,
        historical_days=args.historical_days,
        current_days=args.current_days,
        delta_high=args.delta_high,
        delta_moderate=args.delta_moderate,
        coverage_threshold=args.coverage,
    )

    logger.info("=" * 60)
    logger.info("AGENTE DE DETECCIÓN DE TENDENCIAS")
    logger.info(f"  Límite textos  : {args.limit}")
    logger.info(f"  N tópicos      : {args.n_topics or 'auto'}")
    logger.info(f"  Ventana actual : últimos {args.current_days} días")
    logger.info(f"  Δ alto/mod     : {args.delta_high} / {args.delta_moderate}")
    logger.info(f"  Cobertura mín  : {args.coverage:.0%}")
    logger.info("=" * 60)

    summary = agent.run(limit=args.limit)

    # Mostrar resultados automáticamente
    if summary.get("total_texts", 0) > 0:
        show_results(db, summary["model_run_id"])


if __name__ == "__main__":
    main()
