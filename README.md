# Agentic Reddit Political Analysis

Sistema de analisis politico de Reddit basado en arquitectura agentic con patron ReAct. Detecta sentimiento y tendencias tematicas en el discurso del subreddit r/politics, orquestado por un StateGraph de LangGraph con decisiones condicionales.

## Arquitectura

```
Reddit Data (PRAW / Arctic Shift)
       |
  Preprocessor        Filtra bots, deleted, <10 palabras
       |               Genera: text_for_sentiment, text_for_topics
       |
  Agente Sentimiento   ReAct: Observe -> Reason -> Act -> Record
  (RoBERTa + Gemini)   Decisiones: accepted / cross_validated / rescued / ambiguous
       |
  Agente Tendencias    ReAct: BERTopic + Delta temporal
  (BERTopic + Delta)   Decisiones: emerging / spike / moderate / discarded
       |
  Orquestador          Hay tendencias relevantes?
  (LangGraph)          SI -> Validacion | NO -> Reporte de ausencia
       |
  Agente Validacion    Alertas + Novedad + Gemini Flash
  (Alertas + LLM)      Genera: reporte + word clouds + contexto
       |
  Reporte Final
```

### Agentes ReAct

- **Agente de Sentimiento**: Clasifica comentarios usando RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) con validacion cruzada mediante Gemini 2.5 Flash Lite para casos de confianza intermedia. Textos de baja confianza son "rescatados" por Gemini o marcados como ambiguos.
- **Agente de Tendencias**: Detecta topicos emergentes usando BERTopic y la metrica Delta, que mide cuantas desviaciones estandar se aleja la frecuencia actual de un topico respecto a su comportamiento historico. La ventana de evaluacion se calcula adaptativamente: `W = max(W_stat, W_lifecycle)`.
- **Agente de Validacion**: Genera alertas (criticas/informativas) y produce contexto politico por tendencia usando Gemini Flash.
- **Orquestador (LangGraph)**: Coordina el flujo con decisiones condicionales.

## Corpus

- **Recoleccion historica**: ~90 dias via [Arctic Shift API](https://arctic-shift.photon-reddit.com) (r/politics, Dic 2025 - Mar 2026)
- **Recoleccion en tiempo real**: PRAW para datos recientes
- **Total**: ~203,000 textos analizados

## Resultados

| Metrica | Valor |
|---------|-------|
| Textos analizados | 203,275 |
| Accuracy vs GT (agentic) | 0.7512 |
| F1-macro | 0.5991 |
| Delta accuracy (agentic - pipeline) | +8.02pp |
| Tasa de ambiguedad | 0.98% |
| Coherencia tematica c_v | 0.7815 |
| Estabilidad Jaccard (3 runs) | 0.731 |
| Ventana de evaluacion (adaptativa) | 1.57 dias (37.8h) |
| Topicos detectados (BERTopic) | 379 |
| Tendencias relevantes (filtradas) | 7 |
| Cohen's Kappa (GT manual vs DeepSeek) | 0.9651 |

## Instalacion

```bash
pip install -r requirements.txt
```

Crear `.env` (ver `.env.example`):
```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=tesis/0.1
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...
```

## Uso

### Flujo completo (orquestador)

```bash
# 1. Recoleccion historica (Arctic Shift, una sola vez)
python -m scripts.collect_data --arctic --days 90

# 2. Recoleccion reciente (PRAW)
python -m scripts.collect_data --days 7

# 3. Preprocesamiento
python -m scripts.preprocess_data

# 4. Ejecucion completa (orquestador agentic)
python -m scripts.run_orchestrator

# 5. Evaluacion experimental
python -m scripts.run_evaluation --all

# 6. Interfaz Streamlit
streamlit run app.py
```

### Ejecucion por agentes individuales

```bash
python -m scripts.run_sentiment              # Agente de sentimiento
python -m scripts.run_trends                 # Agente de tendencias
python -m scripts.run_validation             # Agente de validacion
python -m scripts.run_pipeline               # Pipeline tradicional (baseline)
```

### Demo en vivo (defensa)

```bash
# Recolecta ultimos 5 minutos + preprocesa
python -m scripts.collect_data --live --minutes 5

# Ejecutar orquestador completo sobre datos existentes
python -m scripts.run_orchestrator --skip-preprocess

# Abrir dashboard
streamlit run app.py
```

### Inspeccion de datos

```bash
python -m scripts.inspect_sentiment          # Ver clasificaciones
python -m scripts.inspect_trends             # Ver tendencias detectadas
python -m scripts.inspect_ground_truth       # Ver ground truth vs sistema
python -m scripts.inspect_preprocessing      # Ver preprocesamiento
```

## Estructura del Proyecto

```
agentic-reddit-political-analysis/
|-- config/
|   |-- settings.py                 # Configuracion centralizada y credenciales
|
|-- src/
|   |-- collection/
|   |   |-- reddit_client.py        # Cliente PRAW (read-only)
|   |   |-- collector.py            # Recolector tiempo real (PRAW)
|   |   |-- arctic_collector.py     # Recolector historico (Arctic Shift API)
|   |
|   |-- preprocessing/
|   |   |-- text_cleaner.py         # Limpieza de texto (regex, markdown, bots)
|   |   |-- preprocessor.py         # Pipeline de preprocesamiento
|   |
|   |-- agents/
|   |   |-- sentiment/
|   |   |   |-- sentiment_agent.py  # Agente ReAct de sentimiento (RoBERTa + Gemini)
|   |   |
|   |   |-- trends/
|   |   |   |-- trends_agent.py     # Agente ReAct de tendencias (BERTopic + Delta)
|   |   |
|   |   |-- validation/
|   |       |-- validation_agent.py # Agente ReAct de validacion (alertas + LLM)
|   |       |-- report_generator.py # Generador de graficos matplotlib + reportes MD
|   |
|   |-- orchestrator/
|   |   |-- orchestrator.py         # Orquestador LangGraph (StateGraph)
|   |
|   |-- pipeline/
|   |   |-- traditional_pipeline.py # Baseline sin comportamiento agentic
|   |
|   |-- database/
|       |-- models.py               # Esquema SQL (10 tablas)
|       |-- db_manager.py           # CRUD y queries cross-table
|
|-- scripts/
|   |-- collect_data.py             # Recoleccion (PRAW, Arctic, live demo, continua)
|   |-- preprocess_data.py          # Preprocesamiento de textos
|   |-- run_orchestrator.py         # Orquestador LangGraph
|   |-- run_sentiment.py            # Agente de sentimiento standalone
|   |-- run_trends.py               # Agente de tendencias standalone
|   |-- run_validation.py           # Agente de validacion standalone
|   |-- run_pipeline.py             # Pipeline tradicional (baseline)
|   |-- run_evaluation.py           # Evaluacion experimental completa
|   |-- run_comparison.py           # Comparacion agentic vs pipeline
|   |-- label_ground_truth.py       # Etiquetado GT con DeepSeek V3
|   |-- export_manual_sample.py     # Exportar muestra para validacion manual
|   |-- reclassify_with_gemini.py   # Re-clasificacion masiva con Gemini
|   |-- window_analysis.py          # Analisis de ventana temporal optima
|   |-- inspect_sentiment.py        # Inspeccion visual de sentimiento
|   |-- inspect_trends.py           # Inspeccion visual de tendencias
|   |-- inspect_ground_truth.py     # Inspeccion GT vs sistema
|   |-- inspect_preprocessing.py    # Inspeccion de preprocesamiento
|
|-- app.py                          # Interfaz Streamlit (dashboard)
|-- data/                           # Base de datos SQLite
|-- reports/                        # Reportes generados (graficos + markdown)
|-- docs/                           # Diagramas draw.io y PNG
```

## Stack Tecnico

| Tecnologia | Uso |
|------------|-----|
| `praw` | Reddit API (recoleccion en tiempo real) |
| `requests` | Arctic Shift API (recoleccion historica) |
| `transformers` + `torch` | RoBERTa (HuggingFace) |
| `google-genai` | Gemini 2.5 Flash Lite (cross-validacion + contexto) |
| `bertopic` + `sentence-transformers` | Modelado tematico |
| `langgraph` | Orquestacion agentic (StateGraph) |
| `streamlit` | Interfaz web de visualizacion |
| `matplotlib` | Graficos de reportes |
| `SQLite` | Almacenamiento persistente (10 tablas) |
| `loguru` | Logging estructurado |
| `scikit-learn` | Vectorizacion y metricas |
| `scipy` | Curve fitting (ventana adaptativa) |
| `gensim` | Coherencia tematica (evaluacion) |

## Documentacion

- [`DOCUMENTACION_CODIGO.md`](DOCUMENTACION_CODIGO.md) — Documentacion tecnica de todos los modulos