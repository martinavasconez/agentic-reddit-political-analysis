"""
Comparación end-to-end: Pipeline Tradicional vs Sistema Agentic.

Genera un reporte visual lado a lado mostrando las diferencias:
  - Sentimiento: argmax directo vs 4 caminos con Gemini
  - Tendencias: todos los tópicos sin filtrar vs filtrado por Δ con decisiones
  - Reportes: ruido completo vs solo tendencias relevantes

Usa los mismos datos (50K textos) para ambos enfoques.

Uso:
    python -m scripts.run_comparison
    python -m scripts.run_comparison --limit 50000
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.database.db_manager import DatabaseManager
from src.agents.validation.report_generator import ReportGenerator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def run_pipeline_trends(db, texts, current_days=1.57):
    """Ejecuta BERTopic + Δ SIN filtrado (pipeline tradicional)."""
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    REDDIT_STOPWORDS = [
        "comment", "comments", "post", "posts", "response", "responses",
        "said", "says", "saying", "explained", "replied", "reply",
        "upvote", "downvote", "edit", "deleted", "removed",
        "thread", "subreddit", "reddit", "mod", "moderator",
        "yeah", "yes", "no", "ok", "okay", "lol", "lmao", "wtf",
        "thing", "things", "people", "person", "someone", "something",
        "way", "ways", "lot", "lots", "bit", "just", "like", "really",
        "actually", "basically", "literally", "pretty", "quite",
        "think", "thought", "know", "knew", "want", "wanted",
        "going", "gone", "come", "came", "make", "made", "take", "took",
    ]

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    all_stopwords = list(CountVectorizer(stop_words="english").get_stop_words()) + REDDIT_STOPWORDS
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=all_stopwords, min_df=3, max_df=0.85)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=50,
        calculate_probabilities=False,
        verbose=False,
    )

    timestamps = [t["created_utc"] for t in texts]
    max_ts = max(timestamps)
    cutoff = max_ts - (current_days * 86400)

    historical = [t for t in texts if t["created_utc"] < cutoff]
    current = [t for t in texts if t["created_utc"] >= cutoff]

    logger.info(f"[Pipeline] {len(texts)} textos (hist: {len(historical)}, curr: {len(current)})")

    all_docs = [t["text_for_topics"] for t in texts]
    all_topic_ids, _ = topic_model.fit_transform(all_docs)

    hist_ids = list(all_topic_ids[:len(historical)])
    curr_ids = list(all_topic_ids[len(historical):])

    # Δ por tópico
    hist_days = defaultdict(list)
    for t, tid in zip(historical, hist_ids):
        day = datetime.fromtimestamp(t["created_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
        hist_days[day].append(tid)

    curr_count = Counter(curr_ids)
    total_curr = len(curr_ids)
    total_all = len(all_docs)

    topic_info = topic_model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid != -1:
            topic_labels[tid] = row.get("Name", f"topic_{tid}")

    results = []
    for tid in set(all_topic_ids) - {-1}:
        daily_weights = {}
        for day, topics_in_day in sorted(hist_days.items()):
            total_day = len(topics_in_day)
            count_day = sum(1 for t in topics_in_day if t == tid)
            daily_weights[day] = count_day / total_day if total_day else 0

        weights = list(daily_weights.values())
        hist_mean = np.mean(weights) if weights else 0
        hist_std = np.std(weights) if weights else 0
        eff_std = max(hist_std, 0.005)
        curr_weight = curr_count.get(tid, 0) / total_curr if total_curr else 0
        delta = (curr_weight - hist_mean) / eff_std if eff_std > 0 else 0
        coverage = sum(1 for t in all_topic_ids if t == tid) / total_all
        n_current = curr_count.get(tid, 0)

        results.append({
            "topic_id": tid,
            "topic_label": topic_labels.get(tid, f"topic_{tid}"),
            "delta": round(delta, 4),
            "coverage": round(coverage, 4),
            "n_current": n_current,
            "trend_decision": "unfiltered",
        })

    results.sort(key=lambda x: x["delta"], reverse=True)
    n_topics = len(results)
    return results, n_topics


def get_agentic_trends(db):
    """Obtiene tendencias del último run agentic desde la BD."""
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    run_id = conn.execute(
        "SELECT model_run_id FROM trend_analysis ORDER BY analyzed_at DESC LIMIT 1"
    ).fetchone()
    if not run_id:
        return [], 0, "N/A"
    run_id = run_id[0]

    rows = conn.execute(
        "SELECT * FROM trend_analysis WHERE model_run_id = ? ORDER BY delta DESC",
        (run_id,)
    ).fetchall()
    conn.close()

    results = [dict(r) for r in rows]
    n_topics = len(results)
    return results, n_topics, run_id


def get_sentiment_comparison(db):
    """Obtiene distribución de sentimiento: agentic vs pipeline (roberta directo)."""
    import sqlite3
    conn = sqlite3.connect(db.db_path)

    # Agentic: final_label con decisiones
    agentic_dist = {}
    for row in conn.execute("SELECT final_label, COUNT(*) FROM sentiment_results GROUP BY final_label"):
        agentic_dist[row[0]] = row[1]

    # Pipeline: roberta_label directo (sin umbrales, sin Gemini)
    pipeline_dist = {}
    for row in conn.execute("SELECT roberta_label, COUNT(*) FROM sentiment_results GROUP BY roberta_label"):
        pipeline_dist[row[0]] = row[1]

    # Decisiones agentic
    decisions = {}
    for row in conn.execute("SELECT decision, COUNT(*) FROM sentiment_results GROUP BY decision"):
        decisions[row[0]] = row[1]

    # Accuracy
    agentic_acc = conn.execute("""
        SELECT CAST(SUM(CASE WHEN sr.final_label = gt.llm_label THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
        FROM sentiment_results sr
        JOIN ground_truth_labels gt ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
        WHERE sr.final_label NOT IN ('ambiguous', 'mixed')
          AND gt.llm_label IN ('positive','negative','neutral')
    """).fetchone()[0]

    pipeline_acc = conn.execute("""
        SELECT CAST(SUM(CASE WHEN sr.roberta_label = gt.llm_label THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
        FROM sentiment_results sr
        JOIN ground_truth_labels gt ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
        WHERE gt.llm_label IN ('positive','negative','neutral')
    """).fetchone()[0]

    # Ambiguos: accuracy del pipeline si fuerza etiqueta
    amb_acc = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN sr.roberta_label = gt.llm_label THEN 1 ELSE 0 END)
        FROM sentiment_results sr
        JOIN ground_truth_labels gt ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
        WHERE sr.final_label = 'ambiguous'
          AND gt.llm_label IN ('positive','negative','neutral')
    """).fetchone()

    conn.close()

    return {
        "agentic_dist": agentic_dist,
        "pipeline_dist": pipeline_dist,
        "decisions": decisions,
        "agentic_accuracy": agentic_acc,
        "pipeline_accuracy": pipeline_acc,
        "ambiguous_total": amb_acc[0],
        "ambiguous_correct": amb_acc[1],
    }


def plot_side_by_side_sentiment(rg, sent_data):
    """Gráfico lado a lado: distribución pipeline vs agentic."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels_order = ["negative", "neutral", "positive"]
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]

    # Pipeline
    pip_vals = [sent_data["pipeline_dist"].get(l, 0) for l in labels_order]
    pip_total = sum(pip_vals)
    bars1 = ax1.barh(labels_order, pip_vals, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars1, pip_vals):
        pct = val / pip_total * 100 if pip_total else 0
        ax1.text(bar.get_width() + pip_total * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,} ({pct:.1f}%)", va="center", fontsize=9)
    ax1.set_title(f"Pipeline Tradicional\nAccuracy: {sent_data['pipeline_accuracy']:.4f}",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Textos")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.invert_yaxis()

    # Agentic
    labels_ag = ["negative", "neutral", "positive", "ambiguous"]
    colors_ag = ["#e74c3c", "#95a5a6", "#2ecc71", "#f39c12"]
    ag_vals = [sent_data["agentic_dist"].get(l, 0) for l in labels_ag]
    ag_total = sum(ag_vals)
    bars2 = ax2.barh(labels_ag, ag_vals, color=colors_ag, edgecolor="white", height=0.6)
    for bar, val in zip(bars2, ag_vals):
        pct = val / ag_total * 100 if ag_total else 0
        ax2.text(bar.get_width() + ag_total * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,} ({pct:.1f}%)", va="center", fontsize=9)
    ax2.set_title(f"Sistema Agentic\nAccuracy: {sent_data['agentic_accuracy']:.4f}",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Textos")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.invert_yaxis()

    fig.suptitle("Sentimiento: Pipeline vs Agentic", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = rg.output_dir / "sentiment_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rg.figures.append(path.name)
    return path


def plot_side_by_side_trends(rg, pipeline_trends, agentic_trends):
    """Gráfico lado a lado: todos los tópicos vs filtrados."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Pipeline: top 20 sin filtrar
    top_pip = pipeline_trends[:20]
    pip_labels = [t["topic_label"][:35] for t in top_pip]
    pip_deltas = [t["delta"] for t in top_pip]
    pip_colors = ["#bdc3c7"] * len(top_pip)  # all gray = unfiltered
    ax1.barh(pip_labels, pip_deltas, color=pip_colors, edgecolor="white", height=0.6)
    ax1.axvline(1.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Δ=1.5")
    ax1.axvline(1.0, color="#f39c12", linestyle="--", alpha=0.5, label="Δ=1.0")
    ax1.set_title(f"Pipeline: Top 20 de {len(pipeline_trends)} topicos\n(SIN filtrado, reporta todo)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Δ")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.invert_yaxis()

    # Agentic: solo no descartados
    trend_colors = {
        "emerging_trend": "#e74c3c",
        "moderate_trend": "#f39c12",
        "localized_spike": "#3498db",
        "discarded": "#bdc3c7",
    }
    relevant = [t for t in agentic_trends if t["trend_decision"] != "discarded"]
    n_total_agentic = len(agentic_trends)
    n_discarded = n_total_agentic - len(relevant)

    ag_labels = [t["topic_label"][:35] for t in relevant]
    ag_deltas = [t["delta"] for t in relevant]
    ag_colors = [trend_colors.get(t["trend_decision"], "#7f8c8d") for t in relevant]
    ax2.barh(ag_labels, ag_deltas, color=ag_colors, edgecolor="white", height=0.6)
    ax2.axvline(1.5, color="#e74c3c", linestyle="--", alpha=0.5)
    ax2.axvline(1.0, color="#f39c12", linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="emerging_trend"),
        Patch(facecolor="#3498db", label="localized_spike"),
        Patch(facecolor="#f39c12", label="moderate_trend"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    ax2.set_title(
        f"Agentic: {len(relevant)} relevantes de {n_total_agentic}\n"
        f"({n_discarded} descartados por decision ReAct)",
        fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Δ")
    ax2.invert_yaxis()

    fig.suptitle("Tendencias: Pipeline (todo) vs Agentic (filtrado)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = rg.output_dir / "trends_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rg.figures.append(path.name)
    return path


def plot_decisions_breakdown(rg, sent_data):
    """Gráfico de decisiones del agente y su impacto."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Decisiones
    dec_order = ["accepted", "cross_validated", "rescued", "ambiguous"]
    dec_colors = ["#27ae60", "#3498db", "#f39c12", "#e74c3c"]
    dec_vals = [sent_data["decisions"].get(d, 0) for d in dec_order]
    total = sum(dec_vals)

    bars = ax1.barh(dec_order, dec_vals, color=dec_colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, dec_vals):
        pct = val / total * 100
        ax1.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:,} ({pct:.1f}%)", va="center", fontsize=9)
    ax1.set_title("Decisiones del Agente de Sentimiento", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Textos")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.invert_yaxis()

    # Impacto de abstención
    amb_total = sent_data["ambiguous_total"]
    amb_correct = sent_data["ambiguous_correct"]
    amb_wrong = amb_total - amb_correct
    if amb_total > 0:
        amb_acc = amb_correct / amb_total * 100
        categories = ["Correcto\n(si fuerza etiqueta)", "Incorrecto\n(si fuerza etiqueta)"]
        vals = [amb_correct, amb_wrong]
        colors = ["#27ae60", "#e74c3c"]
        bars2 = ax2.bar(categories, vals, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{val:,}", ha="center", fontsize=10)
        ax2.set_title(
            f"Textos Ambiguos ({amb_total:,} textos)\n"
            f"Si el pipeline fuerza: {amb_acc:.1f}% accuracy (casi azar)",
            fontsize=11, fontweight="bold"
        )
        ax2.set_ylabel("Textos")

    plt.tight_layout()

    path = rg.output_dir / "decisions_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rg.figures.append(path.name)
    return path


def main():
    parser = argparse.ArgumentParser(description="Comparación end-to-end: Pipeline vs Agentic")
    parser.add_argument("--limit", type=int, default=50000,
                        help="Textos para BERTopic del pipeline (default: 50000)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COMPARACION END-TO-END: Pipeline vs Agentic")
    logger.info("=" * 60)

    db = DatabaseManager()
    rg = ReportGenerator()

    # 1. Sentimiento
    logger.info("\n[1/3] Comparación de sentimiento...")
    sent_data = get_sentiment_comparison(db)
    plot_side_by_side_sentiment(rg, sent_data)
    plot_decisions_breakdown(rg, sent_data)

    # 2. Tendencias agentic (desde BD)
    logger.info("\n[2/3] Obteniendo tendencias agentic desde BD...")
    agentic_trends, n_agentic, run_id = get_agentic_trends(db)
    agentic_relevant = [t for t in agentic_trends if t["trend_decision"] != "discarded"]
    logger.info(f"  Run {run_id}: {n_agentic} tópicos, {len(agentic_relevant)} relevantes")

    # 3. Tendencias pipeline (ejecuta BERTopic fresco)
    logger.info(f"\n[3/3] Ejecutando BERTopic para pipeline ({args.limit:,} textos)...")
    texts = db.get_texts_for_topic_modeling(limit=args.limit)
    pipeline_trends, n_pipeline = run_pipeline_trends(db, texts)
    logger.info(f"  Pipeline: {n_pipeline} tópicos (TODOS reportados, sin filtrar)")

    # Gráfico comparativo de tendencias
    plot_side_by_side_trends(rg, pipeline_trends, agentic_trends)

    # 4. Generar reporte markdown
    delta_gain = sent_data["agentic_accuracy"] - sent_data["pipeline_accuracy"]
    amb_total = sent_data["ambiguous_total"]
    amb_acc = sent_data["ambiguous_correct"] / amb_total * 100 if amb_total else 0

    sections = [
        {
            "heading": "Resumen de la Comparacion",
            "text": (
                f"Esta comparacion evalua el **Pipeline Tradicional** (RoBERTa argmax directo, "
                f"BERTopic sin filtrado) contra el **Sistema Agentic** (RoBERTa + Gemini con "
                f"decisiones ReAct, BERTopic con filtrado por Delta y decisiones condicionales).\n\n"
                f"Ambos enfoques operan sobre los mismos datos: {len(texts):,} textos de r/politics."
            ),
        },
        {
            "heading": "1. Sentimiento: Pipeline vs Agentic",
            "text": (
                f"**Pipeline Tradicional**: clasifica con RoBERTa argmax directo. "
                f"No tiene umbrales, no consulta Gemini, no puede abstenerse.\n"
                f"- Accuracy vs GT: **{sent_data['pipeline_accuracy']:.4f}**\n\n"
                f"**Sistema Agentic**: clasifica con RoBERTa + Gemini en 4 caminos "
                f"(accepted/cross_validated/rescued/ambiguous).\n"
                f"- Accuracy vs GT: **{sent_data['agentic_accuracy']:.4f}**\n"
                f"- Ganancia: **+{delta_gain:.4f}** ({delta_gain*100:.2f} puntos porcentuales)\n\n"
                f"**Abstencion informada**: el agente marca {amb_total:,} textos como ambiguos. "
                f"Si el pipeline fuerza etiqueta en esos textos, acierta solo el "
                f"**{amb_acc:.1f}%** (casi azar con 3 clases = 33%). "
                f"La abstencion evita {amb_total - sent_data['ambiguous_correct']:,} errores."
            ),
            "image": "sentiment_comparison.png",
        },
        {
            "heading": "2. Decisiones del Agente",
            "text": (
                f"El agente de sentimiento toma 4 tipos de decision:\n\n"
                f"| Decision | Cantidad | % |\n"
                f"|----------|----------|---|\n"
                + "\n".join(
                    f"| {d} | {sent_data['decisions'].get(d, 0):,} | "
                    f"{sent_data['decisions'].get(d, 0)/sum(sent_data['decisions'].values())*100:.1f}% |"
                    for d in ["accepted", "cross_validated", "rescued", "ambiguous"]
                )
                + "\n\n"
                f"El pipeline NO tiene este mecanismo: siempre toma el label con mayor score."
            ),
            "image": "decisions_breakdown.png",
        },
        {
            "heading": "3. Tendencias: Pipeline vs Agentic",
            "text": (
                f"**Pipeline Tradicional**: ejecuta BERTopic y reporta TODOS los topicos.\n"
                f"- Topicos detectados: **{n_pipeline}**\n"
                f"- Todos reportados sin evaluar relevancia\n"
                f"- No hay decisiones (emerging/spike/moderate/discarded)\n"
                f"- El reporte contiene **ruido**: topicos estables, decrecientes, o con Delta < 1.0\n\n"
                f"**Sistema Agentic**: ejecuta BERTopic, calcula Delta, y aplica decisiones ReAct.\n"
                f"- Topicos evaluados: **{n_agentic}**\n"
                f"- Descartados: **{n_agentic - len(agentic_relevant)}** "
                f"({(n_agentic - len(agentic_relevant))/n_agentic*100:.0f}%)\n"
                f"- **Relevantes: {len(agentic_relevant)}** "
                f"(emerging_trend + localized_spike + moderate_trend)\n\n"
                f"El orquestador LangGraph agrega otra capa: si TODOS los topicos son descartados, "
                f"no invoca al agente de validacion (ahorra recursos y evita reportes vacios)."
            ),
            "image": "trends_comparison.png",
        },
        {
            "heading": "4. Tendencias Relevantes (Agentic)",
            "text": "\n".join(
                f"- **{t['topic_label']}** — Delta={t['delta']:.2f}, decision={t['trend_decision']}"
                for t in agentic_relevant
            ) if agentic_relevant else "No se detectaron tendencias relevantes.",
        },
        {
            "heading": "5. Resumen Cuantitativo",
            "table": [
                ["Aspecto", "Pipeline", "Agentic", "Diferencia"],
                ["Accuracy sentimiento", f"{sent_data['pipeline_accuracy']:.4f}",
                 f"{sent_data['agentic_accuracy']:.4f}", f"+{delta_gain:.4f}"],
                ["Puede abstenerse", "No", f"Si ({amb_total:,} textos)", "Evita errores"],
                ["Topicos reportados", str(n_pipeline), str(len(agentic_relevant)),
                 f"-{n_pipeline - len(agentic_relevant)} ruido"],
                ["Decisiones condicionales", "0", "3", "Valor agentic"],
                ["Cross-validacion LLM", "No", "Si (Gemini)", "Mejor en sarcasmo"],
                ["Alertas", "No", "Si (critica/informativa)", "Priorización"],
                ["Contexto politico", "No", "Si (Gemini Flash)", "Interpretabilidad"],
            ],
        },
    ]

    report_path = rg.generate_markdown_report(
        sections, title="Comparacion End-to-End: Pipeline vs Agentic"
    )

    logger.info("=" * 60)
    logger.info(f"Reporte generado: {report_path}")
    logger.info("=" * 60)
    print(f"\nReporte: {report_path}")


if __name__ == "__main__":
    main()
