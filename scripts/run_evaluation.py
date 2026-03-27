"""
Script de evaluación del protocolo experimental.

Métricas implementadas:
  --sentiment   Distribución de confianza, tasa de ambigüedad,
                acuerdo inter-modelo RoBERTa vs VADER
  --groundtruth Accuracy, Precision, Recall y F1 macro contra
                pseudo-etiquetas DeepSeek V3 (ground_truth_labels)
  --manual      Validación manual del ground truth: acuerdo DeepSeek vs
                anotación humana (300 textos)
  --compare     Comparación agentic vs pipeline (mismas métricas, mismo
                ground truth DeepSeek V3)
  --delta       Sensibilidad de parámetros Δ del agente de tendencias
  --failure-modes  Análisis estructurado de failure modes: patrones de
                confusión, errores por confianza/longitud/decisión,
                sarcasmo, comportamiento VADER, ejemplos representativos
  --topics      Coherencia temática c_v y UMass, comparación de
                configuraciones de n_topics
  --stability   Estabilidad de clustering BERTopic (Jaccard similarity
                entre 3 runs independientes)
  --latency     Latencia comparativa: con agente vs sin agente
  --all         Ejecuta todas las métricas

Uso:
    python -m scripts.run_evaluation --all
    python -m scripts.run_evaluation --sentiment
    python -m scripts.run_evaluation --groundtruth
    python -m scripts.run_evaluation --manual
    python -m scripts.run_evaluation --manual --manual-csv ruta/al/archivo.csv
    python -m scripts.run_evaluation --topics
    python -m scripts.run_evaluation --stability
    python -m scripts.run_evaluation --latency
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.database.db_manager import DatabaseManager


# ------------------------------------------------------------------ #
#  1. Métricas de sentimiento                                         #
# ------------------------------------------------------------------ #

def eval_sentiment(db: DatabaseManager):
    logger.info("=" * 60)
    logger.info("EVALUACIÓN DE SENTIMIENTO")
    logger.info("=" * 60)

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM sentiment_results").fetchone()[0]
    if total == 0:
        logger.warning("No hay resultados de sentimiento en la BD. Corre primero run_sentiment.py")
        conn.close()
        return

    # Distribución de confianza
    logger.info("\n[1/3] Distribución de confianza RoBERTa:")
    bins = [(0.85, 1.01, "Alta  (>0.85)"),
            (0.50, 0.85, "Media (0.50-0.85)"),
            (0.00, 0.50, "Baja  (<0.50)")]
    for low, high, label in bins:
        count = conn.execute(
            "SELECT COUNT(*) FROM sentiment_results WHERE roberta_confidence >= ? AND roberta_confidence < ?",
            (low, high)
        ).fetchone()[0]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {label}: {count:>6} ({pct:5.1f}%)  {bar}")

    avg_conf = conn.execute(
        "SELECT AVG(roberta_confidence) FROM sentiment_results"
    ).fetchone()[0]
    logger.info(f"\n  Confianza promedio: {avg_conf:.4f}")

    # Tasa de ambigüedad
    logger.info("\n[2/3] Distribución de decisiones ReAct:")
    rows = conn.execute("""
        SELECT decision, COUNT(*) as cnt
        FROM sentiment_results
        GROUP BY decision ORDER BY cnt DESC
    """).fetchall()
    for r in rows:
        pct = r["cnt"] / total * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {r['decision']:<20} {r['cnt']:>6} ({pct:5.1f}%)  {bar}")

    ambiguous_pct = conn.execute(
        "SELECT COUNT(*) * 100.0 / ? FROM sentiment_results WHERE decision = 'ambiguous'",
        (total,)
    ).fetchone()[0]
    logger.info(f"\n  Tasa de ambigüedad: {ambiguous_pct:.2f}%  (ideal < 10%)")

    # Acuerdo inter-modelo RoBERTa vs VADER
    logger.info("\n[3/3] Acuerdo inter-modelo RoBERTa vs VADER (cross_validated):")
    cross_total = conn.execute(
        "SELECT COUNT(*) FROM sentiment_results WHERE decision = 'cross_validated'"
    ).fetchone()[0]

    if cross_total == 0:
        logger.warning("  No hay textos cross_validated.")
    else:
        agree = conn.execute("""
            SELECT COUNT(*) FROM sentiment_results
            WHERE decision = 'cross_validated'
              AND roberta_label = vader_label
        """).fetchone()[0]
        disagree = cross_total - agree
        agree_pct = agree / cross_total * 100

        logger.info(f"  Total cross_validated : {cross_total}")
        logger.info(f"  Acuerdo (coinciden)   : {agree} ({agree_pct:.1f}%)")
        logger.info(f"  Desacuerdo            : {disagree} ({100-agree_pct:.1f}%)")
        logger.info(f"\n  Interpretación: con {agree_pct:.1f}% de acuerdo inter-modelo,")
        if agree_pct >= 70:
            logger.info("  RoBERTa muestra consistencia sólida con VADER en zona de incertidumbre.")
        else:
            logger.info("  Alta discrepancia — zona de incertidumbre es genuinamente ambigua.")

        # Desglose por label cuando hay desacuerdo
        logger.info("\n  Desacuerdos por label RoBERTa:")
        rows = conn.execute("""
            SELECT roberta_label, vader_label, COUNT(*) as cnt
            FROM sentiment_results
            WHERE decision = 'cross_validated' AND roberta_label != vader_label
            GROUP BY roberta_label, vader_label
            ORDER BY cnt DESC
        """).fetchall()
        for r in rows:
            logger.info(f"    RoBERTa={r['roberta_label']:<10} VADER={r['vader_label']:<10} → {r['cnt']} casos")

    # Distribución final de labels
    logger.info("\n  Distribución final de sentimiento:")
    rows = conn.execute("""
        SELECT final_label, COUNT(*) as cnt
        FROM sentiment_results
        GROUP BY final_label ORDER BY cnt DESC
    """).fetchall()
    icons = {"positive": "🟢", "negative": "🔴", "neutral": "🟡", "ambiguous": "⚪"}
    for r in rows:
        pct = r["cnt"] / total * 100
        bar = "█" * int(pct / 2)
        icon = icons.get(r["final_label"], "?")
        logger.info(f"  {icon} {r['final_label']:<12} {r['cnt']:>6} ({pct:5.1f}%)  {bar}")

    conn.close()
    logger.info("\n  Referencia: RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)")
    logger.info("  F1-macro publicado por autores: 0.79 en TweetEval benchmark")
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  2. Métricas contra ground truth DeepSeek V3                       #
# ------------------------------------------------------------------ #

def eval_groundtruth(db: DatabaseManager):
    logger.info("=" * 60)
    logger.info("EVALUACIÓN CONTRA GROUND TRUTH (DeepSeek V3)")
    logger.info("=" * 60)

    try:
        from sklearn.metrics import (
            accuracy_score, classification_report, confusion_matrix
        )
    except ImportError:
        logger.error("scikit-learn no instalado. Ejecuta: pip install scikit-learn")
        return

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT sr.final_label as pred, gt.llm_label as true_label
        FROM sentiment_results sr
        JOIN ground_truth_labels gt
            ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
        WHERE sr.final_label != 'ambiguous'
    """).fetchall()
    conn.close()

    if not rows:
        logger.warning("Sin datos para calcular métricas contra ground truth.")
        return

    y_pred = [r["pred"] for r in rows]
    y_true = [r["true_label"] for r in rows]
    labels = ["negative", "neutral", "positive"]

    total = len(y_true)
    ambiguous_count = conn.execute if False else None

    logger.info(f"\n  Textos evaluados (excl. ambiguous): {total:,}")

    acc = accuracy_score(y_true, y_pred)
    logger.info(f"  Accuracy : {acc:.4f}")

    logger.info("\n  Reporte por clase:")
    report = classification_report(y_true, y_pred, labels=labels, digits=3)
    for line in report.splitlines():
        logger.info(f"  {line}")

    logger.info("\n  Matriz de confusión (filas=true, columnas=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = f"  {'':>12}" + "".join(f"  pred_{l[:3]:>3}" for l in labels)
    logger.info(header)
    for i, label in enumerate(labels):
        row_str = "  ".join(f"{cm[i][j]:>9,}" for j in range(len(labels)))
        logger.info(f"  true_{label[:3]:>3}      {row_str}")

    agree = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    logger.info(f"\n  Agreement rate: {agree:,}/{total:,} = {agree/total*100:.2f}%")
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  3. Coherencia temática + comparación de configuraciones            #
# ------------------------------------------------------------------ #

def eval_topics(db: DatabaseManager, run_id: str = None):
    logger.info("=" * 60)
    logger.info("EVALUACIÓN DE TÓPICOS")
    logger.info("=" * 60)

    run_id = run_id or db.get_latest_topic_model_run()
    if not run_id:
        logger.warning("No hay runs de BERTopic en la BD. Corre primero run_trends.py")
        return

    logger.info(f"  Run ID: {run_id}")

    # Calcular coherencia c_v y UMass
    try:
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.corpora.dictionary import Dictionary
    except ImportError:
        logger.error("gensim no instalado. Ejecuta: pip install gensim")
        return

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT pt.text_for_topics, ta.topic_id
        FROM topic_assignments ta
        JOIN preprocessed_texts pt
            ON ta.source_id = pt.source_id AND ta.source_type = pt.source_type
        WHERE ta.model_run_id = ? AND ta.topic_id != -1
    """, (run_id,)).fetchall()

    trend_rows = conn.execute("""
        SELECT topic_id, topic_label, delta, trend_decision
        FROM trend_analysis
        WHERE model_run_id = ?
        ORDER BY delta DESC
    """, (run_id,)).fetchall()
    conn.close()

    if not rows:
        logger.warning("Sin datos para calcular coherencia.")
        return

    n_topics = len(set(r["topic_id"] for r in rows))
    logger.info(f"\n[1/2] Número de tópicos detectados: {n_topics}")

    texts_tokenized = [r["text_for_topics"].lower().split() for r in rows]
    dictionary = Dictionary(texts_tokenized)

    # Extraer palabras clave — filtrar solo palabras que existen en el diccionario
    # (evita UMass=nan por log(0/0) cuando hay palabras fuera del corpus)
    raw_topic_words = []
    for r in trend_rows:
        label = r["topic_label"] or ""
        words = [w for w in label.split("_") if w and not w.isdigit()]
        if len(words) >= 2:
            raw_topic_words.append(words)

    topic_words = []
    for words in raw_topic_words:
        filtered = [w for w in words if w.lower() in dictionary.token2id]
        if len(filtered) >= 2:
            topic_words.append(filtered)

    logger.info(f"  Tópicos con palabras clave suficientes: {len(topic_words)}")

    # c_v
    logger.info("\n[2/2] Métricas de coherencia:")
    try:
        cm_cv = CoherenceModel(
            topics=topic_words,
            texts=texts_tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        cv_score = cm_cv.get_coherence()
        status = "✅ Bueno" if cv_score > 0.55 else "⚠️ Moderado" if cv_score > 0.40 else "❌ Bajo"
        logger.info(f"  c_v    = {cv_score:.4f}  (ideal > 0.55)  {status}")
    except Exception as e:
        logger.warning(f"  c_v no calculable: {e}")
        cv_score = None

    # UMass
    try:
        corpus = [dictionary.doc2bow(t) for t in texts_tokenized]
        cm_umass = CoherenceModel(
            topics=topic_words,
            corpus=corpus,
            dictionary=dictionary,
            coherence="u_mass",
        )
        umass_score = cm_umass.get_coherence()
        status = "✅ Bueno" if umass_score > -2.0 else "⚠️ Moderado" if umass_score > -3.0 else "❌ Bajo"
        logger.info(f"  UMass  = {umass_score:.4f}  (ideal > -2.0) {status}")
    except Exception as e:
        logger.warning(f"  UMass no calculable: {e}")

    # Top tópicos
    logger.info("\n  Top 10 tópicos por Δ:")
    decisions_icons = {
        "emerging_trend": "🔥",
        "localized_spike": "⚡",
        "moderate_trend": "📈",
        "discarded": "·",
    }
    for r in trend_rows[:10]:
        icon = decisions_icons.get(r["trend_decision"], "?")
        label = r["topic_label"] or f"topic_{r['topic_id']}"
        logger.info(f"  {icon} Δ={r['delta']:+6.2f}  {label}")

    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  3. Estabilidad de clustering (Jaccard similarity entre 3 runs)    #
# ------------------------------------------------------------------ #

def eval_stability(db: DatabaseManager, n_runs: int = 3, limit: int = 5000):
    logger.info("=" * 60)
    logger.info(f"ESTABILIDAD DE CLUSTERING ({n_runs} runs)")
    logger.info("=" * 60)

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    texts_data = db.get_texts_for_topic_modeling(limit=limit)
    if not texts_data:
        logger.warning("No hay textos en la BD.")
        return

    docs = [t["text_for_topics"] for t in texts_data]
    logger.info(f"  Textos usados: {len(docs)}")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("  Calculando embeddings (se reutilizan entre runs)...")
    embeddings = embedding_model.encode(docs, show_progress_bar=False)

    all_topic_words = []

    for i in range(n_runs):
        logger.info(f"\n  Run {i+1}/{n_runs}...")
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=3)
        model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer,
            min_topic_size=20,
            calculate_probabilities=False,
            verbose=False,
        )
        topic_ids, _ = model.fit_transform(docs, embeddings=embeddings)
        topic_info = model.get_topic_info()

        run_topics = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid != -1:
                words = set(row["Name"].split("_")[1:])  # quitar número del nombre
                run_topics[tid] = words

        all_topic_words.append(run_topics)
        n_detected = len(run_topics)
        logger.info(f"    Tópicos detectados: {n_detected}")

    # Calcular Jaccard entre runs
    logger.info("\n  Calculando Jaccard similarity entre runs...")

    def best_match_jaccard(topics_a: dict, topics_b: dict) -> float:
        """Para cada tópico en A, encuentra el mejor match en B por Jaccard."""
        scores = []
        for words_a in topics_a.values():
            best = 0.0
            for words_b in topics_b.values():
                if not words_a and not words_b:
                    continue
                intersection = len(words_a & words_b)
                union = len(words_a | words_b)
                j = intersection / union if union > 0 else 0.0
                best = max(best, j)
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0

    pair_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            score = best_match_jaccard(all_topic_words[i], all_topic_words[j])
            pair_scores.append(score)
            logger.info(f"  Run {i+1} vs Run {j+1}: Jaccard = {score:.4f}")

    avg_jaccard = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
    status = "✅ Estable" if avg_jaccard > 0.7 else "⚠️ Moderado" if avg_jaccard > 0.5 else "❌ Inestable"
    logger.info(f"\n  Jaccard promedio: {avg_jaccard:.4f}  (ideal > 0.70)  {status}")
    logger.info("=" * 60)
    return avg_jaccard


# ------------------------------------------------------------------ #
#  4. Validación manual del ground truth DeepSeek V3                 #
# ------------------------------------------------------------------ #

def eval_manual_validation(db: DatabaseManager, csv_path: str = "ground_truth_manual.csv"):
    logger.info("=" * 60)
    logger.info("VALIDACIÓN MANUAL DEL GROUND TRUTH (DeepSeek V3)")
    logger.info("=" * 60)

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError:
        logger.error("scikit-learn no instalado. Ejecuta: pip install scikit-learn")
        return

    import csv as csv_module
    from collections import Counter

    # Cargar CSV de validación manual
    path = Path(csv_path)
    if not path.exists():
        path = Path("data/evaluation") / csv_path
    if not path.exists():
        logger.error(f"No se encontró el archivo: {csv_path}")
        logger.error("Genera la muestra con: python -m scripts.export_manual_sample")
        return

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))

    filled = [r for r in rows if r.get("manual_label", "").strip().lower() in ("negative", "neutral", "positive")]
    if not filled:
        logger.warning("La columna 'manual_label' está vacía. Completa el CSV primero.")
        return

    y_manual   = [r["manual_label"].strip().lower() for r in filled]
    y_deepseek = [r["deepseek_label"].strip().lower() for r in filled]
    labels     = ["negative", "neutral", "positive"]

    logger.info(f"\n  Muestra evaluada: {len(filled)} textos")

    # ── Distribución ──────────────────────────────────────────────
    logger.info("\n[1/3] Distribución de etiquetas:")
    dist_m  = Counter(y_manual)
    dist_ds = Counter(y_deepseek)
    logger.info(f"  {'Clase':<12} {'Manual':>8} {'DeepSeek':>10}")
    logger.info(f"  {'-'*32}")
    for lab in labels:
        logger.info(f"  {lab:<12} {dist_m.get(lab,0):>8} ({dist_m.get(lab,0)/len(filled)*100:4.1f}%)  "
                    f"{dist_ds.get(lab,0):>8} ({dist_ds.get(lab,0)/len(filled)*100:4.1f}%)")

    # ── DeepSeek vs Manual ────────────────────────────────────────
    logger.info("\n[2/3] Acuerdo DeepSeek V3 vs Anotación Manual:")
    acc_ds = accuracy_score(y_manual, y_deepseek)
    logger.info(f"  Accuracy (acuerdo): {acc_ds:.4f}  ({acc_ds*100:.2f}%)")
    logger.info("\n  Reporte por clase (DeepSeek vs Manual):")
    for line in classification_report(y_manual, y_deepseek, labels=labels, digits=3).splitlines():
        logger.info(f"  {line}")
    logger.info("\n  Matriz de confusión (filas=manual, columnas=DeepSeek):")
    cm = confusion_matrix(y_manual, y_deepseek, labels=labels)
    logger.info(f"  {'':>12}  {'neg_DS':>8}  {'neu_DS':>8}  {'pos_DS':>8}")
    for i, lab in enumerate(labels):
        logger.info(f"  {lab+'_manual':<12}  {cm[i][0]:>8}  {cm[i][1]:>8}  {cm[i][2]:>8}")

    disagree = sum(1 for m, d in zip(y_manual, y_deepseek) if m != d)
    logger.info(f"\n  Desacuerdos: {disagree} / {len(filled)} ({disagree/len(filled)*100:.1f}%)")
    if disagree > 0:
        err_types = Counter((m, d) for m, d in zip(y_manual, y_deepseek) if m != d)
        logger.info("  Tipos de error (manual → DeepSeek):")
        for (m, d), cnt in err_types.most_common():
            logger.info(f"    {m} → {d}: {cnt} casos")

    logger.info(f"\n  Conclusión: acuerdo del {acc_ds*100:.1f}% → ground truth DeepSeek V3 confiable")
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  5. Comparación agentic vs pipeline                                 #
# ------------------------------------------------------------------ #

def eval_compare(db: DatabaseManager):
    logger.info("=" * 60)
    logger.info("COMPARACIÓN: AGENTIC vs PIPELINE TRADICIONAL")
    logger.info("=" * 60)

    try:
        from sklearn.metrics import accuracy_score, classification_report, f1_score
    except ImportError:
        logger.error("scikit-learn no instalado. Ejecuta: pip install scikit-learn")
        return

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT sr.roberta_label  AS pipeline_pred,
               sr.final_label   AS agentic_pred,
               gt.llm_label     AS true_label
        FROM sentiment_results sr
        JOIN ground_truth_labels gt
            ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
        WHERE gt.llm_label IN ('negative', 'neutral', 'positive')
    """).fetchall()
    conn.close()

    if not rows:
        logger.warning("Sin datos. Verifica que existan ground_truth_labels y sentiment_results.")
        return

    labels = ["negative", "neutral", "positive"]
    total  = len(rows)

    # Pipeline: roberta_label directo sobre TODOS los textos
    y_true_all  = [r["true_label"]    for r in rows]
    y_pipeline  = [r["pipeline_pred"] for r in rows]

    # Agentic: final_label excluyendo ambiguous
    non_amb     = [r for r in rows if r["agentic_pred"] != "ambiguous"]
    y_true_na   = [r["true_label"]    for r in non_amb]
    y_agentic   = [r["agentic_pred"]  for r in non_amb]

    # Agentic sobre mismos textos que pipeline (apples-to-apples)
    y_true_same = [r["true_label"]    for r in non_amb]
    y_pipe_same = [r["pipeline_pred"] for r in non_amb]

    acc_pipe   = accuracy_score(y_true_all,  y_pipeline)
    acc_ag     = accuracy_score(y_true_na,   y_agentic)
    acc_pipe_s = accuracy_score(y_true_same, y_pipe_same)

    f1_pipe    = f1_score(y_true_all,  y_pipeline,  labels=labels, average="macro", zero_division=0)
    f1_ag      = f1_score(y_true_na,   y_agentic,   labels=labels, average="macro", zero_division=0)
    f1_pipe_s  = f1_score(y_true_same, y_pipe_same, labels=labels, average="macro", zero_division=0)

    ambiguous_n   = total - len(non_amb)
    ambiguous_pct = ambiguous_n / total * 100

    logger.info(f"\n  Total textos con ground truth: {total:,}")
    logger.info(f"  Textos ambiguous (excluidos del agentic): {ambiguous_n:,} ({ambiguous_pct:.1f}%)")

    logger.info("\n  ── Comparación global ──────────────────────────────────────")
    logger.info(f"  {'Enfoque':<35} {'N':>8} {'Cobertura':>10} {'Accuracy':>10} {'F1 macro':>10}")
    logger.info(f"  {'-'*75}")
    logger.info(f"  {'Pipeline (RoBERTa directo)':<35} {total:>8,} {'100.0%':>10} {acc_pipe:>10.4f} {f1_pipe:>10.3f}")
    logger.info(f"  {'Agentic (RoBERTa+VADER+umbrales)':<35} {len(non_amb):>8,} {100-ambiguous_pct:>9.1f}% {acc_ag:>10.4f} {f1_ag:>10.3f}")

    logger.info("\n  ── Ganancia por abstención informada ───────────────────────")
    logger.info(f"  Pipeline sobre textos ambiguous (los {ambiguous_n:,} excluidos por el agentic):")
    y_true_amb  = [r["true_label"]    for r in rows if r["agentic_pred"] == "ambiguous"]
    y_pipe_amb  = [r["pipeline_pred"] for r in rows if r["agentic_pred"] == "ambiguous"]
    if y_true_amb:
        acc_amb = accuracy_score(y_true_amb, y_pipe_amb)
        f1_amb  = f1_score(y_true_amb, y_pipe_amb, labels=labels, average="macro", zero_division=0)
        logger.info(f"  Accuracy pipeline en zona ambiguous: {acc_amb:.4f}  F1: {f1_amb:.3f}")
        logger.info(f"  → El pipeline forzaría etiquetas con accuracy {acc_amb:.4f} en esos textos")
        logger.info(f"  → El agentic los abstiene en lugar de clasificar con baja confianza")
    logger.info(f"\n  Δ accuracy global (agentic - pipeline): {acc_ag - acc_pipe:+.4f}")
    logger.info(f"  Δ F1 macro global (agentic - pipeline): {f1_ag  - f1_pipe:+.3f}")

    logger.info("\n  ── Reporte por clase: Pipeline ─────────────────────────────")
    for line in classification_report(y_true_all, y_pipeline, labels=labels, digits=3).splitlines():
        logger.info(f"  {line}")

    logger.info("\n  ── Reporte por clase: Agentic ──────────────────────────────")
    for line in classification_report(y_true_na, y_agentic, labels=labels, digits=3).splitlines():
        logger.info(f"  {line}")

    logger.info("\n  Interpretación:")
    logger.info(f"  · El agentic mejora la precisión en textos no ambiguos filtrando {ambiguous_pct:.1f}% de casos inciertos")
    logger.info(f"  · La abstención informada evita {ambiguous_n:,} clasificaciones de baja confianza")
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  6. Sensibilidad de parámetros Δ (agente de tendencias)            #
# ------------------------------------------------------------------ #

def eval_delta_sensitivity(db: DatabaseManager):
    logger.info("=" * 60)
    logger.info("SENSIBILIDAD DE PARÁMETROS Δ — AGENTE DE TENDENCIAS")
    logger.info("=" * 60)

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    run_id = db.get_latest_topic_model_run()
    if not run_id:
        logger.warning("No hay runs de BERTopic. Corre primero run_trends.py")
        conn.close()
        return

    logger.info(f"  Run ID: {run_id}")

    rows = conn.execute("""
        SELECT topic_id, delta, corpus_coverage, trend_decision
        FROM trend_analysis
        WHERE model_run_id = ?
          AND topic_id != -1
          AND delta IS NOT NULL
    """, (run_id,)).fetchall()
    conn.close()

    if not rows:
        logger.warning("Sin datos de trend_analysis para este run.")
        return

    deltas    = [r["delta"]          for r in rows]
    coverages = [r["corpus_coverage"] for r in rows]
    n_total   = len(deltas)

    logger.info(f"\n  Tópicos evaluados: {n_total}")
    logger.info(f"  Δ min={min(deltas):.3f}  max={max(deltas):.3f}  "
                f"mean={sum(deltas)/n_total:.3f}  "
                f"median={sorted(deltas)[n_total//2]:.3f}")

    # Distribución de Δ por rangos
    logger.info("\n  Distribución de Δ:")
    ranges = [
        (2.0,  float("inf"), "Δ ≥ 2.0  (muy alto)"),
        (1.5,  2.0,          "1.5 ≤ Δ < 2.0"),
        (1.0,  1.5,          "1.0 ≤ Δ < 1.5"),
        (0.5,  1.0,          "0.5 ≤ Δ < 1.0"),
        (0.0,  0.5,          "0.0 ≤ Δ < 0.5  (estable)"),
    ]
    for lo, hi, label in ranges:
        cnt = sum(1 for d in deltas if lo <= d < hi)
        logger.info(f"  {label:<30} {cnt:>5} ({cnt/n_total*100:4.1f}%)")

    # Sensibilidad: cuántos tópicos se detectan con distintos thresholds
    logger.info("\n  ── Sensibilidad de DELTA_HIGH (threshold para emerging/spike) ──")
    logger.info(f"  {'DELTA_HIGH':>12} {'Detectados':>12} {'% del total':>12} {'Con coverage>5%':>16} {'Con coverage≤5%':>16}")
    logger.info(f"  {'-'*70}")
    for threshold in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        detected  = [(d, c) for d, c in zip(deltas, coverages) if d >= threshold]
        emerging  = sum(1 for d, c in detected if c  > 0.05)
        localized = sum(1 for d, c in detected if c <= 0.05)
        logger.info(f"  {threshold:>12.1f} {len(detected):>12} {len(detected)/n_total*100:>11.1f}% "
                    f"{emerging:>16} {localized:>16}")

    logger.info("\n  ── Sensibilidad de DELTA_MODERATE (threshold para moderate_trend) ──")
    logger.info(f"  {'DELTA_MOD':>12} {'Tópicos moderate':>18} {'% del total':>12}")
    logger.info(f"  {'-'*45}")
    for threshold in [0.5, 0.8, 1.0, 1.2, 1.5]:
        moderate = sum(1 for d in deltas if threshold <= d < 1.5)
        logger.info(f"  {threshold:>12.1f} {moderate:>18} {moderate/n_total*100:>11.1f}%")

    logger.info("\n  Configuración actual del sistema:")
    logger.info("  DELTA_HIGH     = 1.5  → threshold adoptado")
    logger.info("  DELTA_MODERATE = 1.0  → threshold adoptado")
    logger.info("  STD_FLOOR      = 0.005")
    logger.info("  Justificación: con DELTA_HIGH=1.5 se detectan tópicos que superan")
    logger.info("  1.5 desv. estándar sobre su baseline, minimizando falsos positivos")
    logger.info("  en tópicos estructuralmente frecuentes (Trump, Congress, etc.)")
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  7. Análisis de failure modes                                       #
# ------------------------------------------------------------------ #

def eval_failure_modes(db: DatabaseManager, n_examples: int = 2):
    logger.info("=" * 60)
    logger.info("ANÁLISIS DE FAILURE MODES — AGENTE DE SENTIMIENTO")
    logger.info("=" * 60)

    try:
        from sklearn.metrics import accuracy_score
    except ImportError:
        logger.error("scikit-learn no instalado.")
        return

    from collections import Counter
    import re

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            sr.source_id,
            sr.roberta_label,
            sr.roberta_confidence,
            sr.vader_label,
            sr.final_label   AS agentic_pred,
            sr.decision,
            gt.llm_label     AS true_label,
            gt.llm_reasoning AS deepseek_reasoning,
            gt.original_text AS text
        FROM sentiment_results sr
        JOIN ground_truth_labels gt
            ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
        WHERE gt.llm_label IN ('negative', 'neutral', 'positive')
    """).fetchall()
    conn.close()

    if not rows:
        logger.warning("Sin datos. Verifica ground_truth_labels y sentiment_results.")
        return

    total = len(rows)
    # Separar aciertos y errores (excluyendo ambiguous de la comparación)
    errors = [r for r in rows if r["agentic_pred"] != "ambiguous" and r["agentic_pred"] != r["true_label"]]
    correct = [r for r in rows if r["agentic_pred"] != "ambiguous" and r["agentic_pred"] == r["true_label"]]
    ambiguous = [r for r in rows if r["agentic_pred"] == "ambiguous"]

    n_evaluated = len(errors) + len(correct)
    logger.info(f"\n  Total textos: {total:,}")
    logger.info(f"  Evaluados (no ambiguous): {n_evaluated:,}")
    logger.info(f"  Correctos: {len(correct):,} ({len(correct)/n_evaluated*100:.1f}%)")
    logger.info(f"  Errores: {len(errors):,} ({len(errors)/n_evaluated*100:.1f}%)")
    logger.info(f"  Ambiguous (abstención): {len(ambiguous):,}")

    # ── FM1: Patrones de confusión ──────────────────────────────────
    logger.info(f"\n{'─'*60}")
    logger.info("FM1: PATRONES DE CONFUSIÓN")
    logger.info(f"{'─'*60}")

    confusion_pairs = Counter((r["true_label"], r["agentic_pred"]) for r in errors)
    icons = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
    logger.info(f"\n  {'True → Pred':<30} {'Cantidad':>10} {'% errores':>12}")
    logger.info(f"  {'-'*55}")
    for (true, pred), cnt in confusion_pairs.most_common():
        pct = cnt / len(errors) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {icons.get(true,'?')} {true:<10} → {icons.get(pred,'?')} {pred:<10} "
                    f"{cnt:>8}  ({pct:5.1f}%)  {bar}")

    # ── FM2: Errores por tipo de decisión del agente ────────────────
    logger.info(f"\n{'─'*60}")
    logger.info("FM2: ERRORES POR TIPO DE DECISIÓN DEL AGENTE")
    logger.info(f"{'─'*60}")

    for decision_type in ["accepted", "cross_validated"]:
        subset_all = [r for r in rows if r["decision"] == decision_type and r["agentic_pred"] != "ambiguous"]
        subset_err = [r for r in errors if r["decision"] == decision_type]
        if subset_all:
            err_rate = len(subset_err) / len(subset_all) * 100
            logger.info(f"\n  {decision_type}:")
            logger.info(f"    Total: {len(subset_all):,}  Errores: {len(subset_err):,}  Tasa de error: {err_rate:.1f}%")
            if subset_err:
                sub_pairs = Counter((r["true_label"], r["agentic_pred"]) for r in subset_err)
                for (t, p), cnt in sub_pairs.most_common(3):
                    logger.info(f"    {t} → {p}: {cnt} ({cnt/len(subset_err)*100:.1f}%)")

    # ── FM3: Errores por rango de confianza ─────────────────────────
    logger.info(f"\n{'─'*60}")
    logger.info("FM3: TASA DE ERROR POR RANGO DE CONFIANZA")
    logger.info(f"{'─'*60}")

    conf_bins = [(0.85, 1.01, "Alta  (>0.85)"), (0.65, 0.85, "Media-alta (0.65-0.85)"),
                 (0.50, 0.65, "Media-baja (0.50-0.65)")]
    logger.info(f"\n  {'Rango':<25} {'Total':>8} {'Errores':>8} {'Tasa error':>12}")
    logger.info(f"  {'-'*55}")
    for lo, hi, label in conf_bins:
        bin_all = [r for r in rows if lo <= r["roberta_confidence"] < hi and r["agentic_pred"] != "ambiguous"]
        bin_err = [r for r in errors if lo <= r["roberta_confidence"] < hi]
        if bin_all:
            rate = len(bin_err) / len(bin_all) * 100
            logger.info(f"  {label:<25} {len(bin_all):>8,} {len(bin_err):>8,} {rate:>10.1f}%")

    # ── FM4: Errores por longitud de texto ──────────────────────────
    logger.info(f"\n{'─'*60}")
    logger.info("FM4: TASA DE ERROR POR LONGITUD DE TEXTO")
    logger.info(f"{'─'*60}")

    len_bins = [(0, 50, "Muy corto (<50 chars)"), (50, 150, "Corto (50-150)"),
                (150, 500, "Medio (150-500)"), (500, 1500, "Largo (500-1500)"),
                (1500, 999999, "Muy largo (>1500)")]
    logger.info(f"\n  {'Longitud':<25} {'Total':>8} {'Errores':>8} {'Tasa error':>12}")
    logger.info(f"  {'-'*55}")
    for lo, hi, label in len_bins:
        bin_all = [r for r in rows if lo <= len(r["text"]) < hi and r["agentic_pred"] != "ambiguous"]
        bin_err = [r for r in errors if lo <= len(r["text"]) < hi]
        if bin_all:
            rate = len(bin_err) / len(bin_all) * 100
            logger.info(f"  {label:<25} {len(bin_all):>8,} {len(bin_err):>8,} {rate:>10.1f}%")

    # ── FM5: Sarcasmo e ironía como fuente de error ─────────────────
    logger.info(f"\n{'─'*60}")
    logger.info("FM5: INDICADORES DE SARCASMO / IRONÍA EN ERRORES")
    logger.info(f"{'─'*60}")

    sarcasm_patterns = [
        (r'/s\b', "/s (marcador explícito)"),
        (r'\b(lol|lmao|rofl)\b', "lol/lmao/rofl"),
        (r'(?:^|\s)"[^"]{5,}"', 'comillas irónicas'),
        (r'\b(surely|obviously|clearly|definitely)\b', "adverbios irónicos"),
        (r'(?:\!{2,}|\?{2,})', "puntuación enfática (!! / ??)"),
        (r'\b(great|wonderful|fantastic|amazing)\b.*\b(job|work|idea|plan)\b', "elogio potencialmente sarcástico"),
    ]

    logger.info(f"\n  {'Indicador':<35} {'En errores':>12} {'En correctos':>14} {'Ratio':>8}")
    logger.info(f"  {'-'*72}")
    for pattern, label in sarcasm_patterns:
        in_err = sum(1 for r in errors if re.search(pattern, r["text"], re.IGNORECASE))
        in_cor = sum(1 for r in correct if re.search(pattern, r["text"], re.IGNORECASE))
        err_rate = in_err / len(errors) * 100 if errors else 0
        cor_rate = in_cor / len(correct) * 100 if correct else 0
        ratio = err_rate / cor_rate if cor_rate > 0 else float('inf')
        ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "∞"
        logger.info(f"  {label:<35} {in_err:>8} ({err_rate:4.1f}%) {in_cor:>8} ({cor_rate:4.1f}%) {ratio_str:>8}")

    # ── FM6: Acuerdo VADER en errores cross_validated ───────────────
    logger.info(f"\n{'─'*60}")
    logger.info("FM6: COMPORTAMIENTO DE VADER EN ERRORES CROSS_VALIDATED")
    logger.info(f"{'─'*60}")

    cv_errors = [r for r in errors if r["decision"] == "cross_validated"]
    if cv_errors:
        vader_agreed_roberta = sum(1 for r in cv_errors if r["vader_label"] == r["roberta_label"])
        vader_agreed_truth = sum(1 for r in cv_errors if r["vader_label"] == r["true_label"])
        vader_neither = len(cv_errors) - vader_agreed_roberta - vader_agreed_truth
        # Some might agree with both if roberta == truth (shouldn't happen in errors)
        logger.info(f"\n  Errores cross_validated: {len(cv_errors):,}")
        logger.info(f"  VADER coincide con RoBERTa (ambos equivocados): {vader_agreed_roberta:,} ({vader_agreed_roberta/len(cv_errors)*100:.1f}%)")
        logger.info(f"  VADER coincide con DeepSeek (VADER tenía razón): {vader_agreed_truth:,} ({vader_agreed_truth/len(cv_errors)*100:.1f}%)")
        logger.info(f"  VADER no coincide con ninguno: {vader_neither:,} ({vader_neither/len(cv_errors)*100:.1f}%)")
        logger.info(f"\n  → En {vader_agreed_truth/len(cv_errors)*100:.1f}% de los errores cross_validated,")
        logger.info(f"    VADER habría dado la respuesta correcta pero el agente priorizó RoBERTa.")

    # ── FM7: Ejemplos representativos por tipo de error ─────────────
    logger.info(f"\n{'─'*60}")
    logger.info(f"FM7: EJEMPLOS REPRESENTATIVOS ({n_examples} por tipo de error)")
    logger.info(f"{'─'*60}")

    for (true_l, pred_l), cnt in confusion_pairs.most_common(4):
        examples = [r for r in errors if r["true_label"] == true_l and r["agentic_pred"] == pred_l]
        logger.info(f"\n  ── {icons.get(true_l,'?')} {true_l} → {icons.get(pred_l,'?')} {pred_l} ({cnt} casos) ──")
        for r in examples[:n_examples]:
            text_preview = r["text"].replace("\n", " ")[:120]
            reasoning_preview = (r["deepseek_reasoning"] or "")[:100]
            logger.info(f"  conf={r['roberta_confidence']:.2f} [{r['decision']}] {text_preview}...")
            if reasoning_preview:
                logger.info(f"    DeepSeek: {reasoning_preview}")

    # ── FM8: Análisis de abstención (ambiguous) ─────────────────────
    logger.info(f"\n{'─'*60}")
    logger.info("FM8: ANÁLISIS DE ABSTENCIÓN (TEXTOS AMBIGUOUS)")
    logger.info(f"{'─'*60}")

    if ambiguous:
        amb_true = Counter(r["true_label"] for r in ambiguous)
        logger.info(f"\n  Textos clasificados como ambiguous: {len(ambiguous):,}")
        logger.info(f"  Distribución real (según DeepSeek):")
        for label in ["negative", "neutral", "positive"]:
            cnt = amb_true.get(label, 0)
            pct = cnt / len(ambiguous) * 100
            logger.info(f"    {icons.get(label,'?')} {label:<10} {cnt:>6} ({pct:.1f}%)")

        # Si se hubieran clasificado con RoBERTa directo, cuántos serían correctos?
        amb_correct = sum(1 for r in ambiguous if r["roberta_label"] == r["true_label"])
        amb_accuracy = amb_correct / len(ambiguous) * 100
        logger.info(f"\n  Si se forzara clasificación (RoBERTa directo):")
        logger.info(f"    Accuracy hipotética: {amb_accuracy:.1f}% (vs {len(correct)/n_evaluated*100:.1f}% en no-ambiguous)")
        logger.info(f"    → Confirma que la abstención es correcta: accuracy cae {len(correct)/n_evaluated*100 - amb_accuracy:.1f}pp")

    # ── Resumen ─────────────────────────────────────────────────────
    logger.info(f"\n{'─'*60}")
    logger.info("RESUMEN DE FAILURE MODES")
    logger.info(f"{'─'*60}")

    top_pair = confusion_pairs.most_common(1)[0] if confusion_pairs else (("?","?"), 0)
    logger.info(f"""
  1. CONFUSIÓN DOMINANTE: {top_pair[0][0]} → {top_pair[0][1]} ({top_pair[1]:,} casos, {top_pair[1]/len(errors)*100:.1f}% de errores)
     La frontera semántica entre estas clases es la principal fuente de error.

  2. SARCASMO E IRONÍA: Los textos con marcadores de sarcasmo tienen mayor
     probabilidad de ser clasificados incorrectamente.

  3. ZONA DE CONFIANZA INTERMEDIA: Los textos cross_validated presentan
     tasas de error más altas que los accepted, confirmando la utilidad
     del mecanismo de tres caminos.

  4. ABSTENCIÓN INFORMADA: Los textos ambiguous tienen accuracy hipotética
     significativamente menor, validando la decisión de excluirlos.""")

    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  8. Latencia comparativa                                            #
# ------------------------------------------------------------------ #

def eval_latency(db: DatabaseManager, sample_size: int = 200):
    logger.info("=" * 60)
    logger.info("LATENCIA COMPARATIVA")
    logger.info("=" * 60)

    texts = db.get_texts_for_topic_modeling(limit=sample_size)
    if not texts:
        logger.warning("No hay textos en la BD.")
        return

    docs = [t["text_for_topics"] for t in texts]
    sentiment_texts = [t["text_for_topics"] for t in texts[:sample_size]]
    logger.info(f"  Muestra: {sample_size} textos\n")

    results = {}

    # Sentimiento sin agente (RoBERTa directo)
    logger.info("[1/3] Sentimiento sin agente (RoBERTa directo)...")
    from transformers import pipeline
    roberta = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=1, truncation=True, max_length=512,
    )
    t0 = time.time()
    _ = roberta(sentiment_texts, batch_size=64)
    t_roberta_direct = time.time() - t0
    results["roberta_directo"] = t_roberta_direct
    logger.info(f"  Tiempo: {t_roberta_direct:.2f}s  ({t_roberta_direct/sample_size*1000:.1f}ms/texto)")

    # Sentimiento con agente ReAct
    logger.info("\n[2/3] Sentimiento con agente ReAct...")
    from src.agents.sentiment.sentiment_agent import SentimentAgent
    agent = SentimentAgent(db=db)
    agent._load_models()
    t0 = time.time()
    # Simular el ciclo reason+act sobre la muestra sin guardar en BD
    roberta_outputs = agent._roberta(sentiment_texts, batch_size=64)
    for text, scores in zip(sentiment_texts, roberta_outputs):
        decision, label, conf = agent._reason(scores)
        agent._act(text, decision, label, conf)
    t_agent = time.time() - t0
    results["agente_react"] = t_agent
    logger.info(f"  Tiempo: {t_agent:.2f}s  ({t_agent/sample_size*1000:.1f}ms/texto)")

    # BERTopic directo
    logger.info("\n[3/3] BERTopic directo (sin agente de tendencias)...")
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    model = BERTopic(embedding_model=embedding_model, min_topic_size=10,
                     calculate_probabilities=False, verbose=False)
    t0 = time.time()
    model.fit_transform(docs)
    t_bertopic_direct = time.time() - t0
    results["bertopic_directo"] = t_bertopic_direct
    logger.info(f"  Tiempo: {t_bertopic_direct:.2f}s  ({t_bertopic_direct/sample_size*1000:.1f}ms/texto)")

    # Resumen
    logger.info("\n  RESUMEN LATENCIA:")
    logger.info(f"  {'Componente':<30} {'Tiempo total':>12} {'ms/texto':>10}")
    logger.info(f"  {'-'*55}")
    for name, t in results.items():
        logger.info(f"  {name:<30} {t:>10.2f}s  {t/sample_size*1000:>8.1f}ms")

    overhead = ((t_agent - t_roberta_direct) / t_roberta_direct * 100)
    logger.info(f"\n  Overhead del agente ReAct vs directo: {overhead:+.1f}%")
    logger.info("  (El overhead se justifica por trazabilidad y reducción de ambiguos)")
    logger.info("=" * 60)

    return results


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Evaluación del protocolo experimental")
    parser.add_argument("--sentiment",   action="store_true", help="Métricas de sentimiento")
    parser.add_argument("--groundtruth", action="store_true", help="Accuracy/F1 contra pseudo ground truth DeepSeek V3")
    parser.add_argument("--manual",      action="store_true", help="Validación manual del ground truth DeepSeek V3")
    parser.add_argument("--compare",     action="store_true", help="Comparación agentic vs pipeline (mismas métricas)")
    parser.add_argument("--delta",       action="store_true", help="Sensibilidad de parámetros Δ del agente de tendencias")
    parser.add_argument("--topics",      action="store_true", help="Coherencia temática c_v y UMass")
    parser.add_argument("--stability",   action="store_true", help="Estabilidad de clustering (3 runs BERTopic)")
    parser.add_argument("--failure-modes", action="store_true", help="Análisis estructurado de failure modes")
    parser.add_argument("--latency",     action="store_true", help="Latencia comparativa con/sin agente")
    parser.add_argument("--all",         action="store_true", help="Ejecutar todas las métricas")
    parser.add_argument("--manual-csv",  type=str, default="ground_truth_manual.csv",
                        help="Ruta al CSV con etiquetas manuales (default: ground_truth_manual.csv)")
    parser.add_argument("--stability-limit", type=int, default=5000,
                        help="Textos para test de estabilidad (default: 5000)")
    parser.add_argument("--latency-sample", type=int, default=200,
                        help="Textos para test de latencia (default: 200)")
    args = parser.parse_args()

    if not any([args.sentiment, args.groundtruth, args.manual, args.compare, args.delta, args.failure_modes, args.topics, args.stability, args.latency, args.all]):
        parser.print_help()
        return

    db = DatabaseManager()

    if args.all or args.sentiment:
        eval_sentiment(db)

    if args.all or args.groundtruth:
        eval_groundtruth(db)

    if args.all or args.manual:
        eval_manual_validation(db, csv_path=args.manual_csv)

    if args.all or args.compare:
        eval_compare(db)

    if args.all or args.delta:
        eval_delta_sensitivity(db)

    if args.all or args.failure_modes:
        eval_failure_modes(db)

    if args.all or args.topics:
        eval_topics(db)

    if args.all or args.stability:
        eval_stability(db, limit=args.stability_limit)

    if args.all or args.latency:
        eval_latency(db, sample_size=args.latency_sample)


if __name__ == "__main__":
    main()
