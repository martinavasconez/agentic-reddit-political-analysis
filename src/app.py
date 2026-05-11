"""
Interfaz Streamlit para el Sistema Agentic de Análisis Político en Reddit.
"""

import json
import re
import sqlite3
from pathlib import Path

import streamlit as st

# ── Configuración de página ──
st.set_page_config(
    page_title="Análisis Político Reddit — Sistema Agentic",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path(__file__).parent.parent / "data" / "reddit_political.db"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


# ── Utilidades ──

@st.cache_resource
def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_report_dirs():
    """Devuelve directorios de reportes ordenados por fecha (más reciente primero)."""
    dirs = sorted(REPORTS_DIR.glob("report_*"), reverse=True)
    return [d for d in dirs if (d / "report.md").exists()]


def parse_report_md(report_path: Path) -> dict:
    """Parsea el report.md en secciones."""
    text = (report_path / "report.md").read_text(encoding="utf-8")

    sections = {}
    current_heading = "intro"
    current_content = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_content:
                sections[current_heading] = "\n".join(current_content).strip()
            current_heading = line[3:].strip()
            current_content = []
        elif line.startswith("# ") and current_heading == "intro":
            sections["title"] = line[2:].strip()
        else:
            current_content.append(line)

    if current_content:
        sections[current_heading] = "\n".join(current_content).strip()

    return sections


def get_images(report_path: Path) -> dict:
    """Devuelve dict de nombre → path para imágenes del reporte."""
    images = {}
    for img in report_path.glob("*.png"):
        images[img.name] = str(img)
    return images


# ── Sidebar ──

st.sidebar.title("Sistema Agentic")
st.sidebar.caption("Análisis de Sentimiento y Tendencias en r/politics")

page = st.sidebar.radio(
    "Navegación",
    ["Reporte", "Métricas Agentic vs Pipeline", "Explorador de Datos", "Arquitectura"],
    index=0,
)

# ── Página: Reporte ──

if page == "Reporte":
    report_dirs = get_report_dirs()

    if not report_dirs:
        st.warning("No hay reportes generados. Ejecuta el orquestador primero.")
        st.stop()

    # Selector de reporte
    report_labels = [d.name.replace("report_", "").replace("_", " @ ") for d in report_dirs]
    selected_idx = st.sidebar.selectbox(
        "Seleccionar reporte",
        range(len(report_dirs)),
        format_func=lambda i: report_labels[i],
    )
    report_path = report_dirs[selected_idx]
    sections = parse_report_md(report_path)
    images = get_images(report_path)

    # Título
    st.title(sections.get("title", "Reporte de Análisis"))

    # Fecha
    intro = sections.get("intro", "")
    date_match = re.search(r"\*Generado: (.+?)\*", intro)
    if date_match:
        st.caption(f"Generado: {date_match.group(1)}")

    # Resumen ejecutivo
    if "Resumen Ejecutivo (generado por LLM)" in sections:
        st.header("Resumen Ejecutivo")
        st.markdown(sections["Resumen Ejecutivo (generado por LLM)"])

    # Hallazgos
    if "Hallazgos del Análisis" in sections:
        st.header("Hallazgos del Análisis")
        findings = sections["Hallazgos del Análisis"]
        # Parsear alertas
        lines = findings.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "ALERTA" in line:
                if "CRITICAL" in line:
                    st.error(line.replace("- ", ""))
                else:
                    st.warning(line.replace("- ", ""))
            elif line.startswith("- "):
                st.markdown(line)

    # Gráficos principales
    st.header("Distribuciones")
    col1, col2 = st.columns(2)
    with col1:
        if "sentiment_distribution.png" in images:
            st.image(images["sentiment_distribution.png"], caption="Distribución de Sentimiento")
    with col2:
        if "confidence_distribution.png" in images:
            st.image(images["confidence_distribution.png"], caption="Distribución de Confianza")

    col3, col4 = st.columns(2)
    with col3:
        if "sentiment_by_topic.png" in images:
            st.image(images["sentiment_by_topic.png"], caption="Sentimiento por Tópico Trending")
    with col4:
        if "trend_deltas.png" in images:
            st.image(images["trend_deltas.png"], caption="Análisis de Tendencias (Delta)")

    # Alertas
    if "Alertas del Sistema" in sections:
        st.header("Alertas del Sistema")
        alerts_text = sections["Alertas del Sistema"]
        for line in alerts_text.split("\n"):
            line = line.strip()
            if "INFORMATIVA" in line:
                st.warning(line.replace("### ", ""))
            elif "CRÍTICA" in line or "CRITICAL" in line:
                st.error(line.replace("### ", ""))
            elif line and not line.startswith("#"):
                st.markdown(line)

    # Tendencias individuales
    st.header("Tendencias Detectadas")
    trend_sections = {k: v for k, v in sections.items() if k.startswith("Tendencia:")}

    for heading, content in trend_sections.items():
        topic_name = heading.replace("Tendencia: ", "")
        with st.expander(f"📊 {topic_name}", expanded=True):
            # Extraer métricas
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("**Δ"):
                    st.markdown(line)
                elif line.startswith("Sentimiento:"):
                    st.markdown(line)
                elif line.startswith("Tópico **nuevo**"):
                    st.info("🆕 Tópico nuevo (sin precedentes en análisis anteriores)")
                elif line.startswith("Relacionado con:"):
                    st.markdown(f"🔗 {line}")
                elif "Alerta:" in line:
                    st.warning(line)

            # Contexto político
            if "### Contexto político" in content:
                ctx = content.split("### Contexto político")[1]
                # Quitar la referencia a imagen al final
                ctx = re.sub(r"!\[.*?\]\(.*?\)", "", ctx).strip()
                st.markdown("**Contexto Político (generado por LLM):**")
                st.markdown(ctx)

            # Word cloud y trend daily
            col_wc, col_td = st.columns(2)
            # Buscar imágenes por prefijo del tópico (ej: "5_christian" → "wordcloud_5_christian")
            # Usar el ID del tópico (primer segmento) para match exacto
            topic_parts = topic_name.split("_")
            topic_id = topic_parts[0] if topic_parts else ""
            # Match: wordcloud_{id}_{siguiente palabra} para evitar que "5" matchee "165"
            topic_prefix = f"{topic_id}_{topic_parts[1]}" if len(topic_parts) > 1 else topic_id

            for img_name, img_path in images.items():
                if "wordcloud" in img_name and f"wordcloud_{topic_prefix}" in img_name:
                    with col_wc:
                        st.image(img_path, caption="Word Cloud")
                    break
            for img_name, img_path in images.items():
                if "trend_daily" in img_name and f"trend_daily_{topic_prefix}" in img_name:
                    with col_td:
                        st.image(img_path, caption="Evolución Temporal")
                    break


# ── Página: Métricas ──

elif page == "Métricas Agentic vs Pipeline":
    st.title("Comparación: Agentic vs Pipeline Tradicional")

    conn = get_db()

    # Obtener datos
    rows = [dict(r) for r in conn.execute("""
        SELECT sr.roberta_label, sr.roberta_confidence, sr.final_label,
               sr.decision, sr.gemini_label,
               gt.llm_label AS true_label
        FROM sentiment_results sr
        JOIN ground_truth_labels gt
            ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
        WHERE gt.llm_label IN ('negative', 'neutral', 'positive')
    """).fetchall()]

    if not rows:
        st.warning("No hay datos de evaluación.")
        st.stop()

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    labels = ["negative", "neutral", "positive"]
    total = len(rows)
    non_amb = [r for r in rows if r["final_label"] != "ambiguous"]
    amb = [r for r in rows if r["final_label"] == "ambiguous"]

    y_true_all = [r["true_label"] for r in rows]
    y_pipe = [r["roberta_label"] for r in rows]
    y_true_ag = [r["true_label"] for r in non_amb]
    y_pred_ag = [r["final_label"] for r in non_amb]

    import pandas as pd
    import numpy as np

    acc_pipe = accuracy_score(y_true_all, y_pipe)
    acc_ag = accuracy_score(y_true_ag, y_pred_ag)
    f1_pipe = f1_score(y_true_all, y_pipe, labels=labels, average="macro", zero_division=0)
    f1_ag = f1_score(y_true_ag, y_pred_ag, labels=labels, average="macro", zero_division=0)

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 1: Resumen general
    # ══════════════════════════════════════════════════════════════
    st.header("1. Resumen General")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Textos Totales", f"{total:,}")
    with col2:
        st.metric("Acc Pipeline", f"{acc_pipe:.2%}")
    with col3:
        st.metric("Acc Agentic", f"{acc_ag:.2%}", delta=f"{acc_ag - acc_pipe:+.2%}")
    with col4:
        st.metric("Abstención", f"{len(amb):,} ({len(amb)/total*100:.1f}%)")

    st.info(
        "**Nota:** La mejora en accuracy (+1.17%) es modesta porque el valor del sistema agentic "
        "no es solo clasificar mejor, sino **saber cuándo es confiable**, **filtrar ruido en tendencias** "
        "y **generar reportes contextualizados**. Las secciones siguientes demuestran cada dimensión."
    )

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 2: Tabla comparativa de métricas
    # ══════════════════════════════════════════════════════════════
    st.header("2. Métricas de Sentimiento")

    metrics_data = []
    for name, avg in [("Accuracy", None), ("F1 Macro", "macro"), ("F1 Weighted", "weighted"),
                      ("Precision Macro", "macro"), ("Recall Macro", "macro")]:
        if name == "Accuracy":
            vp = f"{acc_pipe:.4f}"
            va = f"{acc_ag:.4f}"
            d = f"{acc_ag - acc_pipe:+.4f}"
        elif "F1" in name:
            vp_val = f1_score(y_true_all, y_pipe, labels=labels, average=avg, zero_division=0)
            va_val = f1_score(y_true_ag, y_pred_ag, labels=labels, average=avg, zero_division=0)
            vp, va, d = f"{vp_val:.4f}", f"{va_val:.4f}", f"{va_val - vp_val:+.4f}"
        elif "Precision" in name:
            vp_val = precision_score(y_true_all, y_pipe, labels=labels, average=avg, zero_division=0)
            va_val = precision_score(y_true_ag, y_pred_ag, labels=labels, average=avg, zero_division=0)
            vp, va, d = f"{vp_val:.4f}", f"{va_val:.4f}", f"{va_val - vp_val:+.4f}"
        else:
            vp_val = recall_score(y_true_all, y_pipe, labels=labels, average=avg, zero_division=0)
            va_val = recall_score(y_true_ag, y_pred_ag, labels=labels, average=avg, zero_division=0)
            vp, va, d = f"{vp_val:.4f}", f"{va_val:.4f}", f"{va_val - vp_val:+.4f}"
        metrics_data.append({"Métrica": name, "Pipeline": vp, "Agentic": va, "Delta": d})

    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # F1 por clase
    st.subheader("F1 por Clase")
    f1_class = []
    for l in labels:
        fp = f1_score(y_true_all, y_pipe, labels=[l], average="macro", zero_division=0)
        fa = f1_score(y_true_ag, y_pred_ag, labels=[l], average="macro", zero_division=0)
        f1_class.append({"Clase": l, "Pipeline": f"{fp:.4f}", "Agentic": f"{fa:.4f}", "Delta": f"{fa-fp:+.4f}"})
    st.dataframe(pd.DataFrame(f1_class), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 3: Calibración y estratificación
    # ══════════════════════════════════════════════════════════════
    st.header("3. Calibración: El Agentic Sabe Cuándo Es Confiable")
    st.markdown(
        "El pipeline da **67.11% para todo** sin indicar confiabilidad. "
        "El agentic estratifica en 3 tiers con accuracies muy distintas:"
    )

    tier_data = []
    tier_accs = {}
    for dec in ["accepted", "cross_validated", "rescued", "ambiguous"]:
        subset = [r for r in rows if r["decision"] == dec]
        if subset:
            if dec == "ambiguous":
                acc = accuracy_score([r["true_label"] for r in subset], [r["roberta_label"] for r in subset])
                label = "Ambiguous (excluidos)"
            else:
                acc = accuracy_score([r["true_label"] for r in subset], [r["final_label"] for r in subset])
                label = dec.replace("_", " ").title()
            tier_accs[dec] = acc
            conf_range = {
                "accepted": "> 0.85",
                "cross_validated": "0.50 - 0.85",
                "rescued": "<= 0.50 (Gemini rescató)",
                "ambiguous": "<= 0.50 (sin consenso)",
            }
            tier_data.append({
                "Tier": label,
                "Rango Confianza": conf_range.get(dec, ""),
                "N Textos": f"{len(subset):,}",
                "% Total": f"{len(subset)/total*100:.1f}%",
                "Accuracy": f"{acc:.2%}",
            })

    st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)

    chart_tiers = ["Accepted (>0.85)", "Cross-validated (0.50-0.85)"]
    chart_accs = [tier_accs.get("accepted", 0), tier_accs.get("cross_validated", 0)]
    if "rescued" in tier_accs:
        chart_tiers.append("Rescued (<=0.50)")
        chart_accs.append(tier_accs["rescued"])
    if "ambiguous" in tier_accs:
        chart_tiers.append("Ambiguous (<=0.50)")
        chart_accs.append(tier_accs["ambiguous"])
    chart_tiers.append("Pipeline (todo)")
    chart_accs.append(acc_pipe)

    st.bar_chart(
        pd.DataFrame({
            "Tier": chart_tiers,
            "Accuracy": chart_accs,
        }).set_index("Tier"),
    )

    st.success(
        f"**Interpretación:** Un usuario sabe que las predicciones 'accepted' tienen "
        f"**{tier_accs.get('accepted', 0):.0%}** de accuracy, mientras que el pipeline "
        f"solo puede decir \"{acc_pipe:.0%} en promedio\" sin distinción."
    )

    # SECCIÓN 4 (Predicción selectiva) removida

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 5: Errores evitados
    # ══════════════════════════════════════════════════════════════
    st.header("5. Abstención Informada: Errores Evitados")

    amb_wrong = sum(1 for r in amb if r["roberta_label"] != r["true_label"])
    amb_right = sum(1 for r in amb if r["roberta_label"] == r["true_label"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Textos Ambiguos", f"{len(amb):,}")
    with col2:
        st.metric("Errores Evitados", f"{amb_wrong:,}", help="Clasificaciones incorrectas que el pipeline forzaría")
    with col3:
        st.metric("Tasa Error en Ambiguos", f"{amb_wrong/len(amb)*100:.1f}%")

    st.markdown(
        f"Si el pipeline clasifica los {len(amb):,} textos ambiguos: "
        f"**{amb_wrong:,} errores** + {amb_right:,} aciertos. "
        f"El agentic **evita emitir {amb_wrong:,} etiquetas incorrectas** al reconocer que no puede clasificar con confianza."
    )

    st.warning(
        "**Argumento clave:** La abstención no es 'trampa' — es una decisión informada. "
        "Un sistema que dice 'no sé' cuando acierta solo 44.8% es más útil que uno que adivina mal el 55.2% del tiempo. "
        "Sistemas médicos y financieros usan esta misma estrategia."
    )

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 6: Filtrado de tendencias (signal-to-noise)
    # ══════════════════════════════════════════════════════════════
    st.header("6. Filtrado Inteligente de Tendencias")

    last_model = conn.execute(
        "SELECT model_run_id FROM trend_analysis ORDER BY rowid DESC LIMIT 1"
    ).fetchone()

    if last_model:
        model_id = last_model[0]
        trend_stats = conn.execute("""
            SELECT trend_decision, COUNT(*) as cnt
            FROM trend_analysis WHERE model_run_id = ?
            GROUP BY trend_decision
        """, (model_id,)).fetchall()
        td = {r[0]: r[1] for r in trend_stats}
        total_t = sum(td.values())
        relevant = sum(v for k, v in td.items() if k != "discarded")
        discarded = td.get("discarded", 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tópicos Detectados", total_t)
        with col2:
            st.metric("Tendencias Relevantes", relevant)
        with col3:
            st.metric("Descartados", discarded)
        with col4:
            st.metric("Reducción de Ruido", f"{discarded/total_t*100:.0f}%")

        st.dataframe(pd.DataFrame([
            {"Métrica": "Tópicos reportados al usuario", "Pipeline": f"{total_t} (todos sin filtrar)", "Agentic": f"{relevant} (filtrados por Delta)"},
            {"Métrica": "Signal-to-noise ratio", "Pipeline": f"{relevant}/{total_t} = {relevant/total_t*100:.1f}%", "Agentic": f"{relevant}/{relevant} = 100%"},
            {"Métrica": "Alertas generadas", "Pipeline": "Ninguna (no tiene sistema de alertas)", "Agentic": f"{sum(1 for k,v in td.items() if k in ('emerging_trend','localized_spike'))} informativas"},
            {"Métrica": "Contexto político (LLM)", "Pipeline": "No", "Agentic": "Sí, por cada tendencia"},
            {"Métrica": "Word clouds", "Pipeline": "No", "Agentic": "Sí, por cada tendencia"},
        ]), use_container_width=True, hide_index=True)

        # Decisiones de tendencias
        st.subheader("Distribución de Decisiones del Agente de Tendencias")
        st.dataframe(pd.DataFrame([
            {"Decisión": k, "Cantidad": v, "Porcentaje": f"{v/total_t*100:.1f}%"}
            for k, v in sorted(td.items(), key=lambda x: x[1], reverse=True)
        ]), use_container_width=True, hide_index=True)

        # Top tendencias
        top = conn.execute("""
            SELECT topic_label, delta, trend_decision, trend_reason
            FROM trend_analysis WHERE model_run_id = ? AND trend_decision != 'discarded'
            ORDER BY delta DESC
        """, (model_id,)).fetchall()
        if top:
            st.subheader("Tendencias No Descartadas")
            st.dataframe(pd.DataFrame([{
                "Tópico": t["topic_label"],
                "Delta": f"{t['delta']:.2f}",
                "Decisión": t["trend_decision"],
                "Razón": t["trend_reason"][:100] if t["trend_reason"] else "",
            } for t in top]), use_container_width=True, hide_index=True)

    st.success(
        f"**El pipeline reportaría {total_t} tópicos** al usuario, de los cuales {discarded} son ruido. "
        f"El agentic filtra y reporta solo **{relevant} tendencias reales** con contexto, alertas y word clouds."
    )

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 7: Interpretabilidad y trazabilidad
    # ══════════════════════════════════════════════════════════════
    st.header("7. Interpretabilidad y Trazabilidad")

    interp_data = [
        {"Característica": "Etiqueta de sentimiento", "Pipeline": "Si", "Agentic": "Si"},
        {"Característica": "Confianza por predicción", "Pipeline": "No", "Agentic": "Si"},
        {"Característica": "Tipo de decisión (accepted/cross/ambiguous)", "Pipeline": "No", "Agentic": "Si"},
        {"Característica": "Cross-validación Gemini LLM", "Pipeline": "No", "Agentic": "Si"},
        {"Característica": "Razón por decisión de tendencia", "Pipeline": "No", "Agentic": "Si"},
        {"Característica": "Sistema de alertas (critical/informative)", "Pipeline": "No", "Agentic": "Si"},
        {"Característica": "Resumen ejecutivo (LLM)", "Pipeline": "No", "Agentic": "Si"},
        {"Característica": "Contexto político por tendencia", "Pipeline": "No", "Agentic": "Si"},
        {"Característica": "Ejecución condicional (salta pasos)", "Pipeline": "No", "Agentic": "Si"},
    ]
    st.dataframe(pd.DataFrame(interp_data), use_container_width=True, hide_index=True)

    st.markdown(
        "**Pipeline:** 1 campo por predicción (etiqueta). "
        "**Agentic:** 5 campos por predicción (etiqueta + confianza + decisión + Gemini + acuerdo) "
        "+ razón por tendencia + alertas + contexto LLM."
    )

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 8: Ground Truth
    # ══════════════════════════════════════════════════════════════
    st.header("8. Validación del Ground Truth")

    st.markdown(
        "El ground truth se generó con DeepSeek V3 (pseudo-labeling) y se validó manualmente "
        "con 300 muestras etiquetadas por un anotador humano."
    )

    gt_data = [
        {"Métrica": "Muestras etiquetadas manualmente", "Valor": "300"},
        {"Métrica": "Acuerdo manual vs DeepSeek", "Valor": "98.33% (295/300)"},
        {"Métrica": "Cohen's Kappa", "Valor": "0.9651"},
        {"Métrica": "Escala Landis & Koch", "Valor": "Almost Perfect (> 0.81)"},
        {"Métrica": "Desacuerdos", "Valor": "5 (todos en frontera negative/neutral)"},
        {"Métrica": "Acc agentic vs manual (300 muestras)", "Valor": "0.7207"},
        {"Métrica": "Acc agentic vs DeepSeek (300 muestras)", "Valor": "0.7276"},
        {"Métrica": "Diferencia GT manual vs pseudo", "Valor": "0.0069 (< 1%)"},
    ]
    st.dataframe(pd.DataFrame(gt_data), use_container_width=True, hide_index=True)

    st.success(
        "**La diferencia entre evaluar con etiquetas manuales vs DeepSeek es < 1%.** "
        "El pseudo-labeling no distorsiona las conclusiones. Además, el GT solo se usa "
        "para evaluación, no para entrenamiento — no hay data leakage."
    )

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 9: Ejecución condicional del orquestador
    # ══════════════════════════════════════════════════════════════
    st.header("9. Ejecución Condicional del Orquestador")

    orch_runs = conn.execute("""
        SELECT run_id, status, steps_completed, started_at
        FROM orchestration_runs ORDER BY started_at DESC LIMIT 5
    """).fetchall()

    if orch_runs:
        st.markdown("Últimas ejecuciones del orquestador (LangGraph):")
        for r in orch_runs:
            steps = r["steps_completed"] or "N/A"
            icon = "white_check_mark" if r["status"] == "completed" else "x"
            if "validation" in steps and "no_trends" not in steps:
                st.success(f"**Run {r['run_id']}** | {steps}")
                st.caption("Tendencias detectadas -> Validación + LLM ejecutados")
            elif "no_trends" in steps:
                st.warning(f"**Run {r['run_id']}** | {steps}")
                st.caption("Sin tendencias relevantes -> Validación + LLM SALTADOS (ahorro de recursos)")
            else:
                st.info(f"**Run {r['run_id']}** [{r['status']}] | {steps}")

    st.markdown(
        "**Pipeline tradicional:** ejecuta todos los pasos siempre, sin importar los datos. "
        "**Orquestador agentic:** adapta el flujo según resultados intermedios — "
        "salta sentimiento si no hay textos, salta validación si no hay tendencias."
    )


# ── Página: Explorador ──

elif page == "Explorador de Datos":
    st.title("Explorador de Datos")

    conn = get_db()
    import pandas as pd

    tab1, tab2, tab3 = st.tabs(["Sentimiento", "Tendencias", "Ejecuciones"])

    with tab1:
        st.subheader("Distribución de Sentimiento")
        dist = conn.execute("""
            SELECT final_label, decision, COUNT(*) as cnt
            FROM sentiment_results
            GROUP BY final_label, decision
            ORDER BY cnt DESC
        """).fetchall()
        df = pd.DataFrame([dict(r) for r in dist])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Muestra de textos por decisión
        st.subheader("Muestra de Textos por Decisión")
        dec_filter = st.selectbox("Tipo de decisión", ["accepted", "cross_validated", "rescued", "ambiguous"])
        samples = conn.execute("""
            SELECT pt.original_text, sr.final_label, sr.roberta_label,
                   sr.roberta_confidence, sr.gemini_label, sr.decision
            FROM sentiment_results sr
            JOIN preprocessed_texts pt ON sr.source_id = pt.source_id AND sr.source_type = pt.source_type
            WHERE sr.decision = ?
            ORDER BY RANDOM() LIMIT 10
        """, (dec_filter,)).fetchall()

        for s in samples:
            conf_str = f"{s['roberta_confidence']:.3f}" if s['roberta_confidence'] is not None else "N/A"
            gemini_lab = s['gemini_label'] or "N/A"
            with st.expander(f"[{s['final_label']}] conf={conf_str}"):
                st.write(s["original_text"][:500] if s["original_text"] else "")
                st.caption(
                    f"RoBERTa: {s['roberta_label']} ({conf_str}) | "
                    f"Gemini: {gemini_lab} | "
                    f"Decisión: {s['decision']}"
                )

    with tab2:
        st.subheader("Tendencias Detectadas")
        last_model = conn.execute(
            "SELECT model_run_id FROM trend_analysis ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if last_model:
            trends = conn.execute("""
                SELECT topic_label, delta, trend_decision, trend_reason, corpus_coverage
                FROM trend_analysis
                WHERE model_run_id = ?
                ORDER BY delta DESC
                LIMIT 30
            """, (last_model[0],)).fetchall()
            df_t = pd.DataFrame([dict(t) for t in trends])
            st.dataframe(df_t, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Ejecuciones del Orquestador")
        runs = conn.execute("""
            SELECT run_id, status, steps_completed, started_at, finished_at
            FROM orchestration_runs
            ORDER BY started_at DESC
            LIMIT 10
        """).fetchall()
        df_r = pd.DataFrame([dict(r) for r in runs])
        st.dataframe(df_r, use_container_width=True, hide_index=True)


# ── Página: Arquitectura ──

elif page == "Arquitectura":
    st.title("Arquitectura del Sistema Agentic")

    st.header("Flujo del Sistema")
    st.code("""
    Reddit Data (PRAW / Arctic Shift)
           │
    ┌──────▼──────┐
    │ Preprocessor │  Filtra: bots, deleted, <10 palabras
    │              │  Genera: text_for_sentiment, text_for_topics
    └──────┬──────┘
           │
    ┌──────▼──────────────┐
    │ Agente Sentimiento  │  ReAct: Observe → Reason → Act → Record
    │ (RoBERTa + Gemini)  │  Decisiones: accepted / cross_validated / rescued / ambiguous
    └──────┬──────────────┘
           │
    ┌──────▼──────────────┐
    │ Agente Tendencias   │  ReAct: BERTopic + Δ temporal
    │ (BERTopic + Δ)      │  Decisiones: emerging / spike / moderate / discarded
    └──────┬──────────────┘
           │
    ┌──────▼──────────────┐
    │    Orquestador      │  ¿Hay tendencias relevantes?
    │    (LangGraph)      │  SÍ → Validación | NO → Reporte de ausencia
    └──────┬──────────────┘
           │
    ┌──────▼──────────────┐
    │ Agente Validación   │  Alertas + Novedad + Gemini Flash
    │ (Alertas + LLM)     │  Genera: reporte + word clouds + contexto
    └──────┬──────────────┘
           │
           ▼
       Reporte Final
    """, language=None)

    st.header("Decisiones por Componente")

    st.subheader("Agente de Sentimiento")
    st.markdown("""
    | Confianza RoBERTa | Decisión | Acción |
    |---|---|---|
    | > 0.85 | `accepted` | Acepta RoBERTa directamente |
    | 0.50 - 0.85 | `cross_validated` | Consulta Gemini LLM como segundo validador |
    | ≤ 0.50 | `rescued` / `ambiguous` | Gemini intenta rescatar; si no puede → ambiguo |
    """)

    st.subheader("Agente de Tendencias")
    st.markdown("""
    | Δ | Condición | Decisión |
    |---|---|---|
    | ≥ 1.5 | cobertura > 5% | `emerging_trend` |
    | ≥ 1.5 | cobertura ≤ 5% | `localized_spike` |
    | 1.0 - 1.5 | peso actual > media histórica | `moderate_trend` |
    | 1.0 - 1.5 | peso actual < media histórica | `discarded` |
    | < 1.0 | — | `discarded` |
    """)

    st.subheader("Agente de Validación")
    st.markdown("""
    | Criterio | Nivel de Alerta |
    |---|---|
    | Δ ≥ 3.0 AND negatividad ≥ 70% | **CRÍTICA** |
    | Δ ≥ 2.0 | **INFORMATIVA** |
    | Similaridad < 0.65 vs históricos | Tópico **nuevo** |
    """)

    st.subheader("Orquestador (LangGraph)")
    st.markdown("""
    | Después de... | Condición | Ruta |
    |---|---|---|
    | Preprocesamiento | ¿Hay textos sin analizar? | Sí → Sentimiento / No → Tendencias |
    | Sentimiento | ¿Hay resultados? | Sí → Tendencias / No → Finalizar |
    | Tendencias | ¿Hay tendencias relevantes? | Sí → **Validación** / No → **Reporte ausencia** |
    """)
