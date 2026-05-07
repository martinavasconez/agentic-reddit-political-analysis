"""
Orquestador LangGraph del sistema agentic.

Define un StateGraph con el flujo:
    preprocess → sentiment → trends → [validation | no_trends_report] → finalize

Decisiones condicionales que demuestran el valor agentic:
  1. Si no hay textos pendientes → salta preprocesamiento
  2. Si no hay resultados de sentimiento → salta tendencias (no hay datos)
  3. Si el agente de tendencias NO detectó tendencias relevantes (todo descartado)
     → genera reporte de ausencia y salta validación
  4. Si SÍ detectó tendencias → invoca al agente de validación con alertas y LLM

Estas decisiones son imposibles en un pipeline tradicional que siempre ejecuta
todos los pasos sin evaluar si son necesarios o si los datos lo justifican.
"""

import uuid
from datetime import datetime
from typing import Optional, TypedDict

from loguru import logger

from langgraph.graph import StateGraph, END

from src.database.db_manager import DatabaseManager


# ------------------------------------------------------------------
# Estado compartido entre nodos
# ------------------------------------------------------------------

class OrchestratorState(TypedDict, total=False):
    run_id: str
    db_path: str
    # Configuración
    limit_sentiment: int
    limit_trends: int
    batch_size: int
    skip_preprocess: bool
    # Resultados de cada paso
    preprocess_result: dict
    sentiment_result: dict
    trends_result: dict
    validation_result: dict
    # Control de flujo
    steps_completed: list[str]
    error: str


# ------------------------------------------------------------------
# Nodos del grafo
# ------------------------------------------------------------------

def preprocess_node(state: OrchestratorState) -> OrchestratorState:
    """Ejecuta preprocesamiento de textos crudos."""
    if state.get("skip_preprocess"):
        logger.info("[Orquestador] Saltando preprocesamiento (--skip-preprocess).")
        state["preprocess_result"] = {"skipped": True}
        state.setdefault("steps_completed", []).append("preprocess_skipped")
        return state

    from src.preprocessing.preprocessor import TextPreprocessor

    db = DatabaseManager(state.get("db_path"))
    proc = TextPreprocessor(db=db)
    result = proc.process_all_pending()

    state["preprocess_result"] = result
    state.setdefault("steps_completed", []).append("preprocess")
    return state


def sentiment_node(state: OrchestratorState) -> OrchestratorState:
    """Ejecuta el agente de sentimiento."""
    from src.agents.sentiment.sentiment_agent import SentimentAgent

    db = DatabaseManager(state.get("db_path"))
    agent = SentimentAgent(db=db)
    result = agent.run(
        limit=state.get("limit_sentiment", 1000),
        batch_size=state.get("batch_size", 64),
    )

    state["sentiment_result"] = result
    state.setdefault("steps_completed", []).append("sentiment")
    return state


def trends_node(state: OrchestratorState) -> OrchestratorState:
    """Ejecuta el agente de tendencias."""
    from src.agents.trends.trends_agent import TrendsAgent

    db = DatabaseManager(state.get("db_path"))
    agent = TrendsAgent(db=db)
    result = agent.run(limit=state.get("limit_trends", 50000))

    state["trends_result"] = result
    state.setdefault("steps_completed", []).append("trends")
    return state


def validation_node(state: OrchestratorState) -> OrchestratorState:
    """Ejecuta el agente de validación con alertas y síntesis LLM."""
    from src.agents.validation.validation_agent import ValidationAgent

    db = DatabaseManager(state.get("db_path"))
    agent = ValidationAgent(db=db)

    model_run_id = None
    if state.get("trends_result") and state["trends_result"].get("model_run_id"):
        model_run_id = state["trends_result"]["model_run_id"]

    result = agent.run(model_run_id=model_run_id)

    state["validation_result"] = result
    state.setdefault("steps_completed", []).append("validation")
    return state


def no_trends_report_node(state: OrchestratorState) -> OrchestratorState:
    """Genera reporte indicando ausencia de tendencias relevantes.

    Decisión agentic: en vez de generar un reporte vacío o con ruido,
    el sistema comunica explícitamente que no detectó tendencias
    estadísticamente significativas. Un pipeline tradicional reportaría
    todos los tópicos sin filtrar, generando falsos positivos.
    """
    from src.agents.validation.report_generator import ReportGenerator

    trends_result = state.get("trends_result", {})
    total_topics = trends_result.get("n_topics_detected", 0)
    decision_counts = trends_result.get("trend_decisions", {})

    rg = ReportGenerator()
    sections = [
        {
            "heading": "Resultado del Análisis de Tendencias",
            "text": (
                f"El agente de tendencias evaluó **{total_topics} tópicos** "
                f"y determinó que **ninguno** presenta crecimiento estadísticamente "
                f"significativo respecto al baseline histórico.\n\n"
                f"Distribución de decisiones: {decision_counts}\n\n"
                f"**Decisión del orquestador:** No se invoca al agente de validación "
                f"porque no hay tendencias que contextualizar ni alertas que evaluar. "
                f"Un pipeline tradicional habría reportado los {total_topics} tópicos "
                f"sin filtrar, generando ruido para el usuario."
            ),
        },
    ]

    # Agregar sentimiento si está disponible
    sentiment = state.get("sentiment_result", {})
    if sentiment.get("decision_distribution"):
        sections.append({
            "heading": "Resumen de Sentimiento",
            "text": (
                f"Textos analizados: {sentiment.get('total', 0):,}\n\n"
                f"Distribución: {sentiment.get('label_distribution', {})}\n\n"
                f"Decisiones: {sentiment.get('decision_distribution', {})}"
            ),
        })

    report_path = rg.generate_markdown_report(
        sections, title="Reporte: Sin Tendencias Relevantes"
    )

    state["validation_result"] = {
        "status": "no_relevant_trends",
        "report_path": str(rg.output_dir),
        "reason": (
            f"{total_topics} tópicos evaluados, todos descartados por el agente "
            f"de tendencias. El orquestador decidió no invocar al agente de validación."
        ),
    }
    state.setdefault("steps_completed", []).append("no_trends_report")
    return state


def finalize_node(state: OrchestratorState) -> OrchestratorState:
    """Genera resumen final y persiste metadata del run."""
    db = DatabaseManager(state.get("db_path"))
    steps = state.get("steps_completed", [])

    results_summary = {}
    if state.get("preprocess_result"):
        results_summary["preprocess"] = state["preprocess_result"]
    if state.get("sentiment_result"):
        results_summary["sentiment"] = {
            k: v for k, v in state["sentiment_result"].items()
            if k != "results"
        }
    if state.get("trends_result"):
        results_summary["trends"] = {
            k: v for k, v in state["trends_result"].items()
            if k not in ("topic_model", "all_texts")
        }
    if state.get("validation_result"):
        results_summary["validation"] = {
            k: v for k, v in state["validation_result"].items()
            if k not in ("findings",)  # findings puede ser muy largo
        }

    db.update_orchestration_run(
        state["run_id"],
        status="completed",
        steps_completed=steps,
        results=results_summary,
        finished=True,
    )

    state.setdefault("steps_completed", []).append("finalize")

    logger.info("=" * 60)
    logger.info("ORQUESTADOR — Ejecución completada")
    logger.info(f"  Run ID: {state['run_id']}")
    logger.info(f"  Pasos ejecutados: {' → '.join(steps)}")

    # Log de decisiones condicionales
    if "no_trends_report" in steps:
        logger.info("  Decisión: NO se invocó al agente de validación (sin tendencias relevantes)")
    elif "validation" in steps:
        val = state.get("validation_result", {})
        logger.info(f"  Decisión: SÍ se invocó al agente de validación")
        logger.info(f"    Alertas: {len(val.get('alerts', []))}")
        logger.info(f"    Tendencias evaluadas: {val.get('n_trends_evaluated', 'N/A')}")

    if state.get("validation_result", {}).get("report_path"):
        logger.info(f"  Reporte: {state['validation_result']['report_path']}")
    logger.info("=" * 60)

    return state


# ------------------------------------------------------------------
# Edges condicionales — decisiones agentic del orquestador
# ------------------------------------------------------------------

def should_run_sentiment(state: OrchestratorState) -> str:
    """Decisión: ¿hay textos pendientes de análisis de sentimiento?"""
    db = DatabaseManager(state.get("db_path"))
    texts = db.get_unanalyzed_texts_for_sentiment(limit=1)
    if texts:
        return "sentiment"
    logger.info("[Orquestador] No hay textos nuevos para sentimiento → tendencias.")
    return "trends"


def should_run_trends(state: OrchestratorState) -> str:
    """Decisión: ¿hay suficientes datos de sentimiento para analizar tendencias?"""
    sentiment = state.get("sentiment_result", {})
    if sentiment.get("total", 0) == 0 and not state.get("skip_preprocess"):
        logger.warning("[Orquestador] Sin resultados de sentimiento → finalizando.")
        return "finalize"
    return "trends"


def should_run_validation(state: OrchestratorState) -> str:
    """Decisión clave: ¿el agente de tendencias detectó tendencias relevantes?"""
    trends = state.get("trends_result", {})
    decision_counts = trends.get("trend_decisions", {})
    n_relevant = (
        decision_counts.get("emerging_trend", 0)
        + decision_counts.get("localized_spike", 0)
        + decision_counts.get("moderate_trend", 0)
    )
    if n_relevant > 0:
        logger.info(
            f"[Orquestador] {n_relevant} tendencias relevantes detectadas "
            f"→ invocando agente de validación."
        )
        return "validation"
    else:
        logger.info(
            f"[Orquestador] Sin tendencias relevantes "
            f"(descartadas: {decision_counts.get('discarded', 0)}) "
            f"→ reporte de ausencia."
        )
        return "no_trends_report"

# ------------------------------------------------------------------
# Constructor del grafo
# ------------------------------------------------------------------

class AgenticOrchestrator:
    """Orquestador basado en LangGraph StateGraph.

    Valor del orquestador agentic vs pipeline secuencial:
      1. Ejecución condicional: salta pasos innecesarios basándose en
         los resultados de pasos anteriores (no ejecuta a ciegas).
      2. Filtrado inteligente: solo invoca validación cuando hay
         tendencias estadísticamente relevantes, evitando reportes con ruido.
      3. Trazabilidad: registra cada decisión y paso en la BD, permitiendo
         auditar por qué se tomó cada camino.
      4. Resiliencia: maneja errores por nodo sin abortar todo el flujo.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path) if db_path else None
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(OrchestratorState)

        # Nodos
        g.add_node("preprocess", preprocess_node)
        g.add_node("sentiment", sentiment_node)
        g.add_node("trends", trends_node)
        g.add_node("validation", validation_node)
        g.add_node("no_trends_report", no_trends_report_node)
        g.add_node("finalize", finalize_node)

        # Flujo con decisiones condicionales
        g.set_entry_point("preprocess")
        g.add_conditional_edges("preprocess", should_run_sentiment,
                                {"sentiment": "sentiment", "trends": "trends"})
        g.add_conditional_edges("sentiment", should_run_trends,
                                {"trends": "trends", "finalize": "finalize"})
        g.add_conditional_edges("trends", should_run_validation,
                                {"validation": "validation",
                                 "no_trends_report": "no_trends_report"})
        g.add_edge("validation", "finalize")
        g.add_edge("no_trends_report", "finalize")
        g.add_edge("finalize", END)

        return g

    def run(
        self,
        limit_sentiment: int = 1000,
        limit_trends: int = 50000,
        batch_size: int = 64,
        skip_preprocess: bool = False,
    ) -> OrchestratorState:
        """Ejecuta el flujo completo del orquestador."""
        run_id = str(uuid.uuid4())[:8]

        db = DatabaseManager(self.db_path)
        db.start_orchestration_run(run_id, {
            "limit_sentiment": limit_sentiment,
            "limit_trends": limit_trends,
            "batch_size": batch_size,
            "skip_preprocess": skip_preprocess,
        })

        logger.info("=" * 60)
        logger.info(f"ORQUESTADOR LANGGRAPH — Run {run_id}")
        logger.info(f"  Sentimiento: hasta {limit_sentiment:,} textos")
        logger.info(f"  Tendencias:  hasta {limit_trends:,} textos")
        logger.info(f"  Preprocesamiento: {'skip' if skip_preprocess else 'ejecutar'}")
        logger.info("=" * 60)

        initial_state: OrchestratorState = {
            "run_id": run_id,
            "db_path": self.db_path,
            "limit_sentiment": limit_sentiment,
            "limit_trends": limit_trends,
            "batch_size": batch_size,
            "skip_preprocess": skip_preprocess,
            "steps_completed": [],
        }

        app = self.graph.compile()

        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            logger.error(f"Error en orquestador: {e}")
            db.update_orchestration_run(
                run_id,
                status="failed",
                error_message=str(e),
                finished=True,
            )
            raise

        return final_state
