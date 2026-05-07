"""
Agente ReAct de Validación y Reporte.

Ciclo ReAct:
  Observación  → Consulta sentiment_results + trend_analysis + JOIN cruzado
  Razonamiento → Evalúa severidad y genera alertas (programático)
  Acción       → Invoca LLM (Gemini Flash) para síntesis textual + genera visualizaciones
  Registro     → Persiste reporte y metadata en BD

Decisiones programáticas (el LLM NO decide, solo redacta):
  - Si Δ ≥ 3.0 y polaridad negativa > 70% → Alerta crítica
  - Si Δ ≥ 2.0 sin carga emocional extrema → Alerta informativa
"""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from src.database.db_manager import DatabaseManager
from src.agents.validation.report_generator import ReportGenerator
from config.settings import GEMINI_API_KEY

# Umbrales de alertas (de la planificación, ajustados al corpus)
ALERT_CRITICAL_DELTA = 3.0
ALERT_CRITICAL_NEG_PCT = 0.70
ALERT_INFORMATIVE_DELTA = 2.0


class ValidationAgent:
    """Agente de validación cruzada, alertas y generación de reportes con LLM."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self._gemini_client = None

    def _get_gemini(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        return self._gemini_client

    # ------------------------------------------------------------------
    # ReAct: Observación
    # ------------------------------------------------------------------

    def _observe(self, model_run_id: Optional[str] = None) -> dict:
        """Recopila todos los datos necesarios para el análisis."""
        logger.info("[Observación] Recopilando datos de sentimiento y tendencias...")

        # Sentimiento global
        sentiment_stats = self.db.get_sentiment_stats()

        # Confianzas para histograma
        all_sentiment = self.db.get_all_sentiment_results(limit=500000)
        confidences = [r["final_confidence"] for r in all_sentiment if r["decision"] != "ambiguous"]

        # Tendencias
        if model_run_id is None:
            model_run_id = self.db.get_latest_topic_model_run()

        trends = []
        sentiment_by_topic = {}
        if model_run_id:
            trends = self.db.get_trend_results(model_run_id)
            raw = self.db.get_sentiment_by_trending_topics(model_run_id)
            for row in raw:
                label = row["topic_label"] or f"topic_{row['topic_id']}"
                if label not in sentiment_by_topic:
                    sentiment_by_topic[label] = {}
                sentiment_by_topic[label][row["final_label"]] = row["count"]

        logger.info(
            f"[Observación] Sentimiento: {sentiment_stats['total_analyzed']:,} textos | "
            f"Tendencias: {len(trends)} tópicos evaluados | "
            f"Run: {model_run_id or 'N/A'}"
        )

        return {
            "sentiment_stats": sentiment_stats,
            "confidences": confidences,
            "trends": trends,
            "sentiment_by_topic": sentiment_by_topic,
            "model_run_id": model_run_id,
        }

    # ------------------------------------------------------------------
    # ReAct: Razonamiento (programático — NO usa LLM)
    # ------------------------------------------------------------------

    def _reason(self, observations: dict) -> dict:
        """Evalúa severidad y genera alertas."""
        logger.info("[Razonamiento] Evaluando tendencias...")

        stats = observations["sentiment_stats"]
        dist = stats.get("label_distribution", {})
        decisions = stats.get("decision_distribution", {})
        total = stats["total_analyzed"]
        trends = observations["trends"]
        sent_by_topic = observations["sentiment_by_topic"]

        # --- Hallazgos generales de sentimiento ---
        findings = []
        if dist:
            dominant = max(dist, key=dist.get)
            pct = dist[dominant] / total * 100 if total else 0
            findings.append(
                f"Sentimiento dominante: **{dominant}** ({pct:.1f}%). "
                f"Distribución: " + ", ".join(
                    f"{k}: {v:,} ({v/total*100:.1f}%)" for k, v in sorted(dist.items())
                )
            )
        if decisions:
            findings.append(
                "Decisiones del agente: " +
                ", ".join(f"{k}: {v:,} ({v/total*100:.1f}%)" for k, v in sorted(decisions.items()))
            )
        confs = observations["confidences"]
        if confs:
            findings.append(
                f"Confianza (no ambiguos): media={np.mean(confs):.4f}, mediana={np.median(confs):.4f}"
            )

        # --- Evaluación de cada tendencia no descartada ---
        alerts = []
        trend_evaluations = []
        non_discarded = [t for t in trends if t["trend_decision"] != "discarded"]

        for trend in non_discarded:
            topic_label = trend.get("topic_label", f"topic_{trend['topic_id']}")
            delta = trend["delta"]
            decision = trend["trend_decision"]

            # Sentimiento de este tópico
            topic_sent = sent_by_topic.get(topic_label, {})
            total_t = sum(topic_sent.values()) if topic_sent else 0
            neg_pct = topic_sent.get("negative", 0) / total_t if total_t else 0

            # --- Decisión de alerta ---
            alert_level = None
            if delta >= ALERT_CRITICAL_DELTA and neg_pct >= ALERT_CRITICAL_NEG_PCT:
                alert_level = "critical"
                alerts.append({
                    "level": "critical",
                    "topic": topic_label,
                    "delta": delta,
                    "neg_pct": neg_pct,
                    "reason": (
                        f"Crecimiento abrupto (Δ={delta:.2f} ≥ {ALERT_CRITICAL_DELTA}) "
                        f"con polaridad negativa {neg_pct:.0%} > {ALERT_CRITICAL_NEG_PCT:.0%}"
                    ),
                })
            elif delta >= ALERT_INFORMATIVE_DELTA:
                alert_level = "informative"
                alerts.append({
                    "level": "informative",
                    "topic": topic_label,
                    "delta": delta,
                    "neg_pct": neg_pct,
                    "reason": (
                        f"Tendencia emergente significativa (Δ={delta:.2f} ≥ {ALERT_INFORMATIVE_DELTA})"
                    ),
                })

            # Ventana temporal
            window_start = trend.get("current_window_start", "")
            window_end = trend.get("current_window_end", "")

            trend_evaluations.append({
                "topic_id": trend["topic_id"],
                "topic_label": topic_label,
                "delta": delta,
                "decision": decision,
                "alert_level": alert_level,
                "neg_pct": neg_pct,
                "sentiment_dist": topic_sent,
                "window_start": window_start,
                "window_end": window_end,
                "n_current": trend.get("n_current_texts", 0),
            })

        # Resumen de tendencias
        if non_discarded:
            findings.append(
                f"Tendencias detectadas: {len(non_discarded)} de {len(trends)} tópicos "
                f"({len([t for t in trend_evaluations if t['alert_level'] == 'critical'])} alertas críticas, "
                f"{len([t for t in trend_evaluations if t['alert_level'] == 'informative'])} informativas)"
            )
        if alerts:
            for a in alerts:
                findings.append(f"ALERTA [{a['level'].upper()}]: {a['topic'][:40]} — {a['reason']}")

        logger.info(
            f"[Razonamiento] {len(findings)} hallazgos, "
            f"{len(alerts)} alertas, "
            f"{len(trend_evaluations)} tendencias evaluadas"
        )

        return {
            "findings": findings,
            "alerts": alerts,
            "trend_evaluations": trend_evaluations,
        }

    # ------------------------------------------------------------------
    # ReAct: Acción — LLM para síntesis + visualizaciones
    # ------------------------------------------------------------------

    def _act(self, observations: dict, reasoning: dict) -> dict:
        """Genera síntesis con Gemini y reportes visuales."""
        logger.info("[Acción] Generando reporte visual y síntesis con LLM...")

        rg = ReportGenerator()
        model_run_id = observations.get("model_run_id")

        # --- Visualizaciones ---
        dist = observations["sentiment_stats"].get("label_distribution", {})
        if dist:
            rg.plot_sentiment_distribution(dist)

        confs = observations["confidences"]
        if confs:
            rg.plot_confidence_distribution(confs)

        sent_by_topic = observations["sentiment_by_topic"]
        if sent_by_topic:
            rg.plot_sentiment_by_topic(sent_by_topic)

        trends = observations["trends"]
        if trends:
            trend_data = [
                {"topic_label": t.get("topic_label", f"topic_{t['topic_id']}"),
                 "delta": t["delta"],
                 "trend_decision": t["trend_decision"]}
                for t in trends
            ]
            rg.plot_trend_deltas(trend_data)

        # Daily trends + word clouds para tópicos no descartados
        non_discarded = [t for t in trends if t["trend_decision"] != "discarded"]
        topic_samples = {}  # topic_label -> sample comments for LLM

        for t in non_discarded[:7]:
            topic_label = t.get("topic_label", f"topic_{t['topic_id']}")
            topic_id = t["topic_id"]

            # Daily trend plot
            if t.get("daily_weights_json"):
                daily = json.loads(t["daily_weights_json"]) if isinstance(t["daily_weights_json"], str) else t["daily_weights_json"]
                if daily:
                    rg.plot_daily_trend(daily, topic_label)

            # Word cloud + sample comments from actual texts
            if model_run_id:
                topic_texts = self.db.get_texts_for_topic(topic_id, model_run_id, limit=200)
                if topic_texts:
                    texts_for_wc = [t_row["cleaned_text"] for t_row in topic_texts if t_row["cleaned_text"]]
                    if texts_for_wc:
                        rg.plot_wordcloud(texts_for_wc, topic_label)

                    # Sample de comentarios representativos para el LLM
                    import random
                    sample = random.sample(topic_texts, min(15, len(topic_texts)))
                    topic_samples[topic_label] = [
                        {
                            "text": s["cleaned_text"][:300],
                            "sentiment": s.get("final_label", "unknown"),
                        }
                        for s in sample if s["cleaned_text"]
                    ]

        # --- Síntesis textual con Gemini ---
        llm_synthesis = self._generate_llm_synthesis(observations, reasoning, topic_samples)

        # --- Generar reporte Markdown ---
        sections = [
            {
                "heading": "Resumen Ejecutivo (generado por LLM)",
                "text": llm_synthesis.get("executive_summary", "No disponible."),
            },
            {
                "heading": "Hallazgos del Análisis",
                "text": "\n\n".join(f"- {f}" for f in reasoning["findings"]),
            },
            {
                "heading": "Distribución de Sentimiento",
                "text": f"Total analizado: {observations['sentiment_stats']['total_analyzed']:,} textos.",
                "image": "sentiment_distribution.png" if dist else None,
            },
            {
                "heading": "Distribución de Confianza",
                "text": "Histograma de confianza RoBERTa (excluyendo textos ambiguos).",
                "image": "confidence_distribution.png" if confs else None,
            },
        ]

        if sent_by_topic:
            sections.append({
                "heading": "Sentimiento por Tópico Trending",
                "text": "Distribución de sentimiento en los tópicos identificados como tendencia.",
                "image": "sentiment_by_topic.png",
            })

        if trends:
            sections.append({
                "heading": "Análisis de Tendencias (Delta)",
                "text": f"{len(trends)} tópicos evaluados, {len(non_discarded)} no descartados.",
                "image": "trend_deltas.png",
            })

        # Sección de alertas
        if reasoning["alerts"]:
            alert_text = ""
            for a in reasoning["alerts"]:
                icon = "CRITICA" if a["level"] == "critical" else "INFORMATIVA"
                alert_text += (
                    f"### [{icon}] {a['topic']}\n\n"
                    f"{a['reason']}\n\n"
                )
            sections.append({
                "heading": "Alertas del Sistema",
                "text": alert_text,
            })

        # Sección por tendencia: contexto LLM + word cloud
        for e in reasoning["trend_evaluations"]:
            topic_label = e["topic_label"]
            safe_name = topic_label[:30].replace(" ", "_").replace("/", "_")
            wc_file = f"wordcloud_{safe_name}.png"

            # Contexto del LLM
            ctx_text = ""
            for ctx in llm_synthesis.get("trend_contexts", []):
                if ctx["topic"] == topic_label:
                    ctx_text = ctx["context"]
                    break

            trend_section_text = f"**Δ = {e['delta']:.2f}** | Decisión: {e['decision']}"
            if e["alert_level"]:
                trend_section_text += f" | Alerta: **{e['alert_level'].upper()}**"
            trend_section_text += f"\n\nSentimiento: {e['sentiment_dist']} (negatividad: {e['neg_pct']:.0%})"
            if ctx_text:
                trend_section_text += f"\n\n### Contexto político\n\n{ctx_text}"

            sections.append({
                "heading": f"Tendencia: {topic_label[:50]}",
                "text": trend_section_text,
                "image": wc_file if (rg.output_dir / wc_file).exists() else None,
            })

        report_path = rg.generate_markdown_report(sections, title="Reporte de Validación Agentic")
        logger.info(f"[Acción] Reporte generado: {report_path}")

        return {
            "report_path": str(rg.output_dir),
            "report_file": str(report_path),
            "llm_synthesis": llm_synthesis,
        }

    def _generate_llm_synthesis(self, observations: dict, reasoning: dict,
                               topic_samples: dict = None) -> dict:
        """Invoca Gemini Flash para generar síntesis textual del reporte."""
        trend_evals = reasoning["trend_evaluations"]
        alerts = reasoning["alerts"]
        stats = observations["sentiment_stats"]
        topic_samples = topic_samples or {}

        # Construir prompt con las decisiones ya tomadas
        trend_descriptions = []
        for e in trend_evals:
            desc = (
                f"- Tópico: {e['topic_label']}\n"
                f"  Δ={e['delta']:.2f}, Decisión: {e['decision']}, "
                f"Alerta: {e['alert_level'] or 'ninguna'}\n"
                f"  Sentimiento: {e['sentiment_dist']}, Negatividad: {e['neg_pct']:.0%}\n"
                f"  Ventana temporal: {e['window_start']} a {e['window_end']}\n"
                f"  Textos en ventana actual: {e['n_current']}"
            )
            # Agregar muestra de comentarios reales
            samples = topic_samples.get(e["topic_label"], [])
            if samples:
                desc += "\n  COMENTARIOS REALES DE REDDIT (muestra):"
                for s in samples[:10]:
                    desc += f"\n    [{s['sentiment']}] \"{s['text'][:200]}\""
            trend_descriptions.append(desc)

        prompt = f"""Eres un analista experto en política estadounidense. Tu tarea es generar un reporte de análisis basado en datos del discurso político en Reddit (r/politics), que es un subreddit enfocado exclusivamente en política de Estados Unidos.

IMPORTANTE: Todo el análisis debe estar contextualizado en la política de EE.UU. (Congreso, Presidente, legislación federal, política estatal, elecciones, agencias federales, etc.).

DATOS DE SENTIMIENTO:
- Total de textos analizados: {stats['total_analyzed']:,}
- Distribución: {stats.get('label_distribution', {})}
- Decisiones del agente: {stats.get('decision_distribution', {})}

TENDENCIAS DETECTADAS ({len(trend_evals)} tendencias no descartadas):
{chr(10).join(trend_descriptions) if trend_descriptions else 'Ninguna tendencia significativa detectada.'}

ALERTAS ACTIVADAS ({len(alerts)}):
{chr(10).join(f"- [{a['level'].upper()}] {a['topic']}: {a['reason']}" for a in alerts) if alerts else 'Ninguna alerta activada.'}

INSTRUCCIONES:
1. Genera un RESUMEN EJECUTIVO (3-5 párrafos) que:
   - Sintetice el estado del discurso político estadounidense en Reddit durante este período
   - Destaque las tendencias más relevantes y su significado en el contexto político de EE.UU.
   - Si hay alertas, explica su importancia para el panorama político doméstico
   - Usa lenguaje académico pero accesible, en español

2. Para CADA tendencia detectada, genera un párrafo de CONTEXTO que:
   - Analice los COMENTARIOS REALES proporcionados para entender de qué hablan los usuarios
   - Explique qué eventos de la política estadounidense de esas fechas explican la tendencia (legislación, decisiones ejecutivas, escándalos, conflictos partidistas, etc.)
   - Explique por qué el sentimiento tiene la polaridad observada en el contexto político de EE.UU.
   - Sea específico: menciona actores políticos, leyes, agencias o eventos concretos cuando sea posible

Responde SOLO en formato JSON válido (sin markdown, sin ```). Escapa comillas dobles dentro de strings con \\". No uses saltos de línea literales dentro de strings JSON, usa \\n. Estructura exacta:
{{
  "executive_summary": "texto del resumen ejecutivo",
  "trend_contexts": [
    {{"topic": "nombre EXACTO del tópico tal como aparece arriba", "context": "párrafo de contexto"}}
  ]
}}"""

        try:
            client = self._get_gemini()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            # Parsear respuesta JSON
            text = response.text.strip()
            # Limpiar markdown code blocks si los hay
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                elif "```" in text:
                    text = text[:text.rfind("```")]
            text = text.strip()

            result = json.loads(text)
            logger.info(f"[LLM] Síntesis generada: {len(result.get('executive_summary', ''))} chars, "
                       f"{len(result.get('trend_contexts', []))} contextos")
            return result

        except json.JSONDecodeError as je:
            logger.warning(f"[LLM] JSON malformado, intentando reparación: {je}")
            try:
                import re
                # Intentar extraer campos con regex como fallback
                exec_match = re.search(
                    r'"executive_summary"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)',
                    text, re.DOTALL,
                )
                executive = exec_match.group(1) if exec_match else ""
                executive = executive.replace('\\"', '"').replace("\\n", "\n")

                # Extraer trend_contexts
                contexts = []
                ctx_pattern = re.finditer(
                    r'\{\s*"topic"\s*:\s*"([^"]+)"\s*,\s*"context"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    text, re.DOTALL,
                )
                for m in ctx_pattern:
                    contexts.append({
                        "topic": m.group(1),
                        "context": m.group(2).replace('\\"', '"').replace("\\n", "\n"),
                    })

                if executive:
                    logger.info(f"[LLM] Reparación exitosa: {len(executive)} chars, {len(contexts)} contextos")
                    return {"executive_summary": executive, "trend_contexts": contexts}
                raise ValueError("No se pudo extraer executive_summary")
            except Exception as repair_err:
                logger.error(f"[LLM] Reparación fallida: {repair_err}")
                return {
                    "executive_summary": (
                        f"Error al generar síntesis con LLM: {je}\n\n"
                        "Los datos programáticos del análisis se presentan a continuación."
                    ),
                    "trend_contexts": [],
                }

        except Exception as e:
            logger.error(f"[LLM] Error al generar síntesis con Gemini: {e}")
            return {
                "executive_summary": (
                    f"Error al generar síntesis con LLM: {e}\n\n"
                    "Los datos programáticos del análisis se presentan a continuación."
                ),
                "trend_contexts": [],
            }

    # ------------------------------------------------------------------
    # ReAct: Registro
    # ------------------------------------------------------------------

    def _record(self, observations: dict, reasoning: dict, action_result: dict) -> int:
        """Persiste metadata del reporte en BD."""
        report_data = {
            "model_run_id": observations.get("model_run_id"),
            "report_path": action_result["report_path"],
            "n_sentiment_analyzed": observations["sentiment_stats"]["total_analyzed"],
            "n_trends_analyzed": len(observations["trends"]),
            "summary": {
                "label_distribution": observations["sentiment_stats"].get("label_distribution"),
                "decision_distribution": observations["sentiment_stats"].get("decision_distribution"),
                "n_trending": len([t for t in observations["trends"]
                                   if t["trend_decision"] != "discarded"]),
                "alerts": reasoning["alerts"],
            },
        }
        report_id = self.db.insert_validation_report(report_data)
        logger.info(f"[Registro] Metadata guardada (id={report_id}).")
        return report_id

    # ------------------------------------------------------------------
    # Ciclo principal
    # ------------------------------------------------------------------

    def run(self, model_run_id: Optional[str] = None) -> dict:
        """Ejecuta el ciclo ReAct completo del agente de validación."""
        logger.info("=" * 60)
        logger.info("AGENTE DE VALIDACIÓN — Ciclo ReAct")
        logger.info("=" * 60)
        observations = self._observe(model_run_id)

        if observations["sentiment_stats"]["total_analyzed"] == 0:
            logger.warning("No hay datos de sentimiento. Ejecute primero el agente de sentimiento.")
            return {"status": "no_data"}

        reasoning = self._reason(observations)
        action_result = self._act(observations, reasoning)
        report_id = self._record(observations, reasoning, action_result)

        logger.info("=" * 60)
        logger.info(f"Reporte: {action_result['report_path']}")
        logger.info(f"Alertas: {len(reasoning['alerts'])}")
        logger.info("=" * 60)

        return {
            "status": "completed",
            "report_id": report_id,
            "report_path": action_result["report_path"],
            "findings": reasoning["findings"],
            "alerts": reasoning["alerts"],
            "n_trends_evaluated": len(reasoning["trend_evaluations"]),
        }
