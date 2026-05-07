# Documentación del Código

## Índice

1. [Visión General](#1-vision-general)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Configuración (`config/settings.py`)](#3-configuracion)
4. [Base de Datos](#4-base-de-datos)
5. [Módulo de Recolección](#5-modulo-de-recoleccion)
6. [Módulo de Preprocesamiento](#6-modulo-de-preprocesamiento)
7. [Agente de Sentimiento](#7-agente-de-sentimiento)
8. [Agente de Tendencias](#8-agente-de-tendencias)
9. [Agente de Validación](#9-agente-de-validacion)
10. [Orquestador LangGraph](#10-orquestador-langgraph)
11. [Pipeline Tradicional (Baseline)](#11-pipeline-tradicional)
12. [Evaluación del Protocolo Experimental](#12-evaluacion)
13. [Interfaz Streamlit](#13-interfaz-streamlit)
14. [Scripts de Ejecución](#14-scripts-de-ejecucion)
15. [Flujo Completo de Datos](#15-flujo-completo)
16. [Parámetros Configurables](#16-parametros-configurables)

---

## 1. Visión General

Este proyecto implementa una **arquitectura agentic** para el análisis de sentimiento y detección de tendencias en el discurso político de Reddit. Los agentes siguen el patrón **ReAct** (Reason + Act): observan datos, razonan sobre ellos y toman decisiones basadas en umbrales.

### Componentes implementados

| Componente | Descripción | Estado |
|-----------|-------------|--------|
| Recolección histórica | Arctic Shift API — 90 días de datos históricos uniformes | ✅ |
| Recolección en tiempo real | PRAW — posts más recientes | ✅ |
| Preprocesamiento | Limpieza y normalización para RoBERTa y BERTopic | ✅ |
| Agente de Sentimiento | RoBERTa + Gemini LLM con patrón ReAct | ✅ |
| Agente de Tendencias | BERTopic + cálculo de Δ temporal | ✅ |
| Agente de Validación | Alertas + contexto LLM | ✅ |
| Orquestador LangGraph | Coordinación con decisiones condicionales | ✅ |
| Pipeline Tradicional | Baseline de comparación (RoBERTa directo) | ✅ |
| Evaluación experimental | c_v, UMass, Jaccard, baselines, ablación | ✅ |
| Interfaz Streamlit | Reportes, métricas, explorador de datos | ✅ |

---

## 2. Estructura del Proyecto

```
agentic-reddit-political-analysis/
├── .env                          # Credenciales Reddit API (no se sube a git)
├── .env.example                  # Plantilla de credenciales
├── requirements.txt              # Dependencias Python
├── README.md                     # Descripción general del proyecto
├── DOCUMENTACION_CODIGO.md       # Documentación técnica detallada
├── config/
│   └── settings.py               # Configuración centralizada
├── data/
│   ├── reddit_political.db       # Base de datos SQLite (no se sube a git)
│   └── evaluation/
│       ├── evaluation_metrics.json  # Métricas del protocolo experimental
│       └── labeled_dataset.csv      # Ground truth generado con DeepSeek V3
├── docs/
│   ├── diagrama_recoleccion.drawio           # Flujo de recolección PRAW + Arctic Shift
│   ├── diagrama_preprocesamiento.drawio      # Pipeline de preprocesamiento
│   ├── diagrama_decision_sentimiento.drawio  # Árbol de decisión del agente de sentimiento
│   └── modelo_entidad_relacion.drawio        # Esquema relacional de la BD
├── scripts/
│   ├── collect_data.py              # Recolección de datos (todos los modos)
│   ├── preprocess_data.py           # Preprocesamiento de textos
│   ├── run_sentiment.py             # Agente de sentimiento
│   ├── run_trends.py                # Agente de tendencias
│   ├── run_evaluation.py            # Protocolo experimental completo
│   ├── label_ground_truth.py        # Etiquetado automático con DeepSeek V3
│   ├── evaluate_ground_truth.py     # Evaluación de métricas contra ground truth
│   ├── run_orchestrator.py           # Ejecución del orquestador LangGraph
│   ├── run_pipeline.py              # Ejecución del pipeline tradicional (baseline)
│   ├── run_validation.py            # Ejecución del agente de validación
│   ├── reclassify_with_gemini.py    # Re-clasificación masiva con Gemini (one-shot)
│   ├── window_analysis.py           # Análisis de half-life y ventana adaptativa
│   ├── export_manual_sample.py      # Exportar muestra para etiquetado manual
│   ├── inspect_sentiment.py         # Inspección visual: texto + clasificación
│   ├── inspect_trends.py            # Inspección visual: tópicos + textos
│   ├── inspect_ground_truth.py      # Comparación ground truth vs RoBERTa
│   ├── show_preprocessing_examples.py  # Ejemplos de transformación textual
│   └── test_sentiment.py            # Verificación rápida de 100 clasificaciones
├── src/
│   ├── collection/
│   │   ├── reddit_client.py      # Conexión a Reddit API (PRAW)
│   │   ├── collector.py          # Recolección via PRAW
│   │   └── arctic_collector.py   # Recolección histórica via Arctic Shift
│   ├── database/
│   │   ├── models.py             # Esquema SQL (8 tablas)
│   │   └── db_manager.py         # Operaciones CRUD
│   ├── preprocessing/
│   │   ├── text_cleaner.py       # Limpieza de texto con regex
│   │   └── preprocessor.py       # Pipeline de preprocesamiento
│   ├── orchestrator/
│   │   └── orchestrator.py       # Orquestador LangGraph StateGraph
│   ├── pipeline/
│   │   └── traditional_pipeline.py  # Pipeline tradicional (baseline)
│   ├── reporting/
│   │   └── report_generator.py   # Generador de gráficos y reportes .md
│   └── agents/
│       ├── sentiment/
│       │   └── sentiment_agent.py  # Agente ReAct de sentimiento (RoBERTa + Gemini)
│       ├── trends/
│       │   └── trends_agent.py     # Agente ReAct de tendencias
│       └── validation/
│           └── validation_agent.py # Agente de validación (alertas + LLM)
```

---

## 3. Configuración

**Archivo**: `config/settings.py`

Centraliza toda la configuración. Carga credenciales de Reddit desde `.env`.

### Variables de entorno (`.env`)

| Variable | Descripción |
|----------|-------------|
| `REDDIT_CLIENT_ID` | Client ID de la app registrada en Reddit |
| `REDDIT_CLIENT_SECRET` | Client Secret |
| `REDDIT_USER_AGENT` | Identificador del agente (default: `tesis/0.1`) |

### Parámetros principales

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `TARGET_SUBREDDITS` | `["politics"]` | Subreddits objetivo |
| `DEFAULT_COLLECTION_DAYS` | 7 | Ventana de recolección PRAW |
| `POSTS_PER_SUBREDDIT` | 500 | Máximo de posts por subreddit (PRAW) |
| `COMMENTS_PER_POST` | 100 | Máximo de comentarios por post (PRAW) |
| `RATE_LIMIT_SLEEP` | 1 | Segundos entre requests al API |
| `MIN_WORD_COUNT` | 10 | Mínimo palabras para texto válido |
| `MAX_TEXT_LENGTH` | 10000 | Máximo caracteres (trunca si supera) |
| `ROBERTA_MAX_TOKENS` | 512 | Límite de tokens para RoBERTa |
| `BERTOPIC_MIN_WORDS` | 15 | Mínimo palabras para BERTopic |

---

## 4. Base de Datos

### 4.1 Esquema

**Archivo**: `src/database/models.py`

Define 8 tablas SQLite:

#### `posts`
Posts extraídos de Reddit.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | TEXT PK | ID único del post en Reddit |
| `subreddit` | TEXT | Nombre del subreddit |
| `title` | TEXT | Título del post |
| `selftext` | TEXT | Cuerpo del post (vacío si es link) |
| `author` | TEXT | Autor (null si fue eliminado) |
| `score` | INTEGER | Score (upvotes - downvotes) |
| `upvote_ratio` | REAL | Ratio de upvotes (0.0 a 1.0) |
| `num_comments` | INTEGER | Total de comentarios |
| `created_utc` | REAL | Timestamp Unix de creación |
| `url` | TEXT | URL del post o enlace |
| `is_self` | INTEGER | 1 si es self post, 0 si es link |
| `permalink` | TEXT | Link permanente en Reddit |
| `collected_at` | TEXT | Timestamp ISO de recolección |

#### `comments`
Comentarios extraídos de cada post.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | TEXT PK | ID único del comentario |
| `post_id` | TEXT FK | ID del post padre |
| `subreddit` | TEXT | Nombre del subreddit |
| `body` | TEXT | Texto del comentario |
| `author` | TEXT | Autor |
| `score` | INTEGER | Score del comentario |
| `created_utc` | REAL | Timestamp Unix de creación |
| `parent_id` | TEXT | ID del padre (post o comentario) |
| `is_root` | INTEGER | 1 si es comentario directo al post |
| `depth` | INTEGER | Profundidad en el árbol de comentarios |
| `controversiality` | INTEGER | Indicador de controversia de Reddit |
| `collected_at` | TEXT | Timestamp ISO de recolección |

#### `preprocessed_texts`
Textos limpios listos para los modelos. Cada fila tiene dos versiones del texto.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | INTEGER PK | ID autoincremental |
| `source_id` | TEXT | ID del post o comentario original |
| `source_type` | TEXT | `'post'` o `'comment'` |
| `subreddit` | TEXT | Nombre del subreddit |
| `original_text` | TEXT | Texto original sin modificar |
| `cleaned_text` | TEXT | Texto con limpieza base |
| `text_for_sentiment` | TEXT | Optimizado para RoBERTa |
| `text_for_topics` | TEXT | Optimizado para BERTopic |
| `word_count` | INTEGER | Palabras del texto limpio |
| `created_utc` | REAL | Fecha del contenido original |
| `processed_at` | TEXT | Timestamp de procesamiento |
| `is_valid` | INTEGER | 1 si tiene >= 10 palabras |

#### `sentiment_results`
Resultados del Agente de Sentimiento con trazabilidad completa de cada decisión ReAct.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | INTEGER PK | ID autoincremental |
| `preprocessed_text_id` | INTEGER FK | Referencia a `preprocessed_texts` |
| `source_id` | TEXT | ID del post o comentario |
| `source_type` | TEXT | `'post'` o `'comment'` |
| `subreddit` | TEXT | Nombre del subreddit |
| `roberta_label` | TEXT | Predicción de RoBERTa: `positive/negative/neutral` |
| `roberta_confidence` | REAL | Confianza de RoBERTa (0.0 a 1.0) |
| `decision` | TEXT | Decisión ReAct: `accepted/cross_validated/rescued/ambiguous` |
| `final_label` | TEXT | Etiqueta final: `positive/negative/neutral/ambiguous` |
| `final_confidence` | REAL | Confianza final (puede tener boost si hay acuerdo) |
| `gemini_label` | TEXT | Etiqueta de Gemini LLM (solo si decision != 'accepted') |
| `analyzed_at` | TEXT | Timestamp de análisis |

#### `topic_assignments`
Asignación de tópico por texto, resultado de BERTopic.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | INTEGER PK | ID autoincremental |
| `preprocessed_text_id` | INTEGER FK | Referencia a `preprocessed_texts` |
| `source_id` | TEXT | ID del post o comentario |
| `source_type` | TEXT | `'post'` o `'comment'` |
| `subreddit` | TEXT | Nombre del subreddit |
| `created_utc` | REAL | Fecha del texto original |
| `topic_id` | INTEGER | ID del tópico (-1 = outlier) |
| `topic_label` | TEXT | Palabras clave del tópico (ej: `"0_trump_tariff_trade"`) |
| `topic_probability` | REAL | Probabilidad de asignación |
| `model_run_id` | TEXT | UUID del run de BERTopic |
| `assigned_at` | TEXT | Timestamp de asignación |

#### `trend_analysis`
Resultados del análisis de tendencias por tópico, con la decisión ReAct y métricas Δ.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | INTEGER PK | ID autoincremental |
| `model_run_id` | TEXT | UUID del run de BERTopic |
| `topic_id` | INTEGER | ID del tópico |
| `topic_label` | TEXT | Palabras clave del tópico |
| `current_weight` | REAL | Peso del tópico en ventana actual |
| `historical_mean` | REAL | Media histórica del peso |
| `historical_std` | REAL | Desv. estándar histórica (antes del floor) |
| `effective_std` | REAL | Desv. estándar usada (con STD_FLOOR aplicado) |
| `delta` | REAL | Δ = (current - mean) / effective_std |
| `current_window_start` | TEXT | Fecha inicio ventana actual |
| `current_window_end` | TEXT | Fecha fin ventana actual |
| `historical_window_start` | TEXT | Fecha inicio ventana histórica |
| `historical_window_end` | TEXT | Fecha fin ventana histórica |
| `n_current_texts` | INTEGER | Textos en ventana actual con este tópico |
| `n_historical_texts` | INTEGER | Textos históricos con este tópico |
| `corpus_coverage` | REAL | % del corpus total cubierto por el tópico |
| `trend_decision` | TEXT | `emerging_trend/localized_spike/moderate_trend/discarded` |
| `trend_reason` | TEXT | Razón textual de la decisión |
| `daily_weights_json` | TEXT | JSON con pesos diarios `{"2026-02-15": 0.12, ...}` |
| `analyzed_at` | TEXT | Timestamp del análisis |

### 4.2 Gestor de BD

**Archivo**: `src/database/db_manager.py`

La clase `DatabaseManager` encapsula toda la interacción con SQLite.

#### Configuración de conexión
```python
conn.row_factory = sqlite3.Row      # Acceso a columnas por nombre
conn.execute("PRAGMA journal_mode=WAL")  # Mejor concurrencia
conn.execute("PRAGMA foreign_keys=ON")   # Habilitar foreign keys
```

#### Métodos por módulo

**Posts y comentarios:**
- `insert_post(post_data)` → `bool`: `INSERT OR IGNORE`, retorna True si era nuevo
- `insert_posts_batch(posts)` → `int`: Inserta múltiples posts en una transacción
- `insert_comments_batch(comments)` → `int`: Inserta múltiples comentarios
- `get_unprocessed_comments(limit=5000)` → `list[dict]`: LEFT JOIN para encontrar comentarios sin preprocesar
- `get_unprocessed_posts(limit=5000)` → `list[dict]`: Similar para posts con selftext

**Sentimiento:**
- `get_unanalyzed_texts_for_sentiment(limit)` → `list[dict]`: LEFT JOIN para textos sin análisis de sentimiento
- `insert_sentiment_batch(results)` → `int`: `INSERT OR IGNORE` batch de resultados
- `get_sentiment_stats(subreddit)` → `dict`: Métricas agregadas (distribución, confianza promedio, % ambiguos)

**Tópicos y tendencias:**
- `get_texts_for_topic_modeling(limit)` → `list[dict]`: Textos válidos con timestamps para BERTopic
- `insert_topic_assignments_batch(assignments)` → `int`: Guarda asignaciones de tópico
- `insert_trend_analysis_batch(trends)` → `int`: Guarda análisis de tendencias
- `get_trend_results(model_run_id, decision_filter)` → `list[dict]`: Resultados del agente de tendencias
- `get_latest_topic_model_run()` → `str`: Retorna el `model_run_id` más reciente

---

## 5. Módulo de Recolección

### 5.1 Cliente Reddit (PRAW)

**Archivo**: `src/collection/reddit_client.py`

Crea una instancia de PRAW en modo **read-only**. Solo se necesitan credenciales de aplicación (no usuario/contraseña) porque solo se leen datos públicos.

### 5.2 Recolector PRAW

**Archivo**: `src/collection/collector.py`

La clase `RedditCollector` extrae datos usando la API oficial de Reddit.

**Limitación importante**: La API de Reddit solo devuelve los ~1,000 posts más recientes en sus endpoints de listado (`new`, `hot`, `top`). Para r/politics esto equivale a ~6-7 días. No puede acceder a datos históricos más antiguos.

#### `collect_subreddit(subreddit_name, days, max_posts, max_comments_per_post)`

Itera por tres métodos de ordenamiento: `new` → `hot` → `top(month)`.
- `new`: Posts más recientes. Hace `break` al encontrar posts fuera del rango temporal.
- `hot`: Posts populares actualmente. Filtra por fecha con `continue`.
- `top(month)`: Top 1000 del mes — cubre 30 días pero solo los más votados.

Usa un `set()` para deduplicación en memoria. Para cada post llama a `_collect_comments()` y espera `RATE_LIMIT_SLEEP` segundos.

### 5.3 Recolector Arctic Shift

**Archivo**: `src/collection/arctic_collector.py`

**Arctic Shift** es un archivo público de Reddit que almacena todos los posts y comentarios desde 2005. Permite consultas por rango de fechas exacto — sin las limitaciones de la API oficial.

**API base**: `https://arctic-shift.photon-reddit.com/api`
- `GET /posts/search?subreddit=X&after=EPOCH&before=EPOCH&limit=100`
- `GET /comments/search?link_id=POST_ID&limit=100`

**Ventajas sobre PRAW para datos históricos:**
- Acceso a cualquier fecha sin límite de 1,000 posts
- Distribución temporal uniforme (100 posts/día consistentes)
- Sin autenticación requerida
- Datos con ~24h de retraso (no sirve para tiempo real)

#### `collect_historical(subreddit, days=30)`

Para cada uno de los últimos N días:
1. Llama a `_get_posts_for_day()` con los timestamps del día
2. Para cada post llama a `_get_comments_for_post()` (hasta 200 comentarios)
3. Filtra posts sin título y respuestas automáticas de moderadores
4. Inserta en BD (idempotente — `INSERT OR IGNORE` evita duplicados)

#### Paginación
Cada request devuelve máximo 100 resultados. Si hay más, se repite la query con `after=último_created_utc` hasta agotar los resultados del día.

#### Idempotencia
Si se ejecuta dos veces sobre el mismo rango de fechas, los posts ya existentes se ignoran (`INSERT OR IGNORE`). El campo `new_posts_inserted` en el resultado indica cuántos eran realmente nuevos.

---

## 6. Módulo de Preprocesamiento

### 6.1 Limpiador de Texto

**Archivo**: `src/preprocessing/text_cleaner.py`

La clase `TextCleaner` implementa tres niveles de limpieza con regex compiladas.

#### `clean_base(text, url_replace="", mention_replace="")`
Limpieza común aplicada antes de ambas versiones:
1. Markdown links → extrae solo texto visible: `[texto](url)` → `texto`
2. URLs → reemplaza con `url_replace`
3. Links de Reddit → elimina `/r/subreddit`, `/u/usuario`
4. Menciones → reemplaza con `mention_replace`
5. Formato markdown → elimina `**`, `*`, `` ` ``, `~~`
6. Entidades HTML → reemplaza `&amp;`, etc.
7. Unicode → normaliza a NFKC
8. Puntuación repetida → `!!!!` → `!`
9. Espacios → normaliza múltiples espacios a uno

#### `clean_for_sentiment(text)` — Para RoBERTa
Aplica `clean_base(url_replace="http", mention_replace="@user")`.

Decisiones fundamentadas en el paper del modelo:
- **NO hace lowercase**: RoBERTa fue entrenado con texto natural, lowercase es destructivo
- **NO elimina números**: Pueden tener carga emocional ("$2 trillion!")
- **URLs → `http`**: Placeholder esperado por `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Menciones → `@user`**: Placeholder esperado por el modelo

#### `clean_for_topics(text)` — Para BERTopic
Aplica `clean_base()` eliminando URLs y menciones completamente.
- Preserva caso original (Trump, NATO son semánticamente relevantes)
- Preserva números y nombres propios
- Elimina emojis

#### `is_bot_content(text)`
Detecta mensajes de bots y auto-moderadores. Filtra patrones como `"I am a bot"`, `"this action was performed automatically"`, `"please contact the moderators"`.

### 6.2 Preprocesador

**Archivo**: `src/preprocessing/preprocessor.py`

#### `process_all_pending()` → `dict`
Método principal. Obtiene comentarios y posts sin preprocesar, los procesa y guarda en batch. Retorna estadísticas (procesados, válidos, filtrados).

#### Pipeline de filtros para cada texto:
1. Vacío → `is_valid=False` (se inserta igual para marcar como procesado)
2. `[deleted]` o `[removed]` → `is_valid=False`
3. Contenido de bot → `is_valid=False`
4. > 10,000 caracteres → trunca (no descarta)
5. Queda vacío después de limpieza → `is_valid=False`
6. Genera `text_for_sentiment` y `text_for_topics`
7. `word_count >= MIN_WORD_COUNT (10)` → `is_valid = True`

**Decisión de diseño clave**: Todos los comentarios — incluso los filtrados — se insertan en `preprocessed_texts` con `is_valid=0`. Esto es fundamental para que `get_unprocessed_comments()` (que usa `LEFT JOIN ... WHERE pt.id IS NULL`) no los vuelva a encontrar. Sin esta corrección, el loop de preprocesamiento se quedaba atascado procesando los mismos ~162K comentarios filtrados indefinidamente.

Para posts: combina `"{title}. {selftext}"` porque el título aporta contexto.

---

## 7. Agente de Sentimiento

**Archivo**: `src/agents/sentiment/sentiment_agent.py`

Implementa el patrón **ReAct** (Observación → Razonamiento → Acción → Registro) para clasificación de sentimiento.

### Modelos utilizados
- **RoBERTa**: `cardiffnlp/twitter-roberta-base-sentiment-latest` — modelo fine-tuneado en Twitter para sentimiento. F1-macro publicado = 0.79 en TweetEval.
- **Gemini 2.5 Flash Lite**: LLM de Google usado como segundo validador. Entiende contexto político, sarcasmo e ironía mejor que modelos basados en léxico.

### Umbrales configurables
```python
HIGH_CONF_THRESHOLD = 0.85   # Por encima: acepta directamente
LOW_CONF_THRESHOLD  = 0.50   # Por debajo: intenta rescatar con Gemini
MID_CONF_THRESHOLD  = 0.65   # Desempate en zona media cuando hay desacuerdo
```

### Ciclo ReAct

#### `_observe(limit)` — Observación
Consulta `get_unanalyzed_texts_for_sentiment()` — LEFT JOIN para obtener textos sin análisis previo. Garantiza idempotencia.

#### `_reason(roberta_scores)` — Razonamiento
Analiza el output de RoBERTa y decide:

| Condición | Decisión |
|-----------|----------|
| `conf > 0.85` | `accepted` — RoBERTa es suficientemente seguro |
| `0.50 < conf ≤ 0.85` | `needs_cross_validation` — necesita validación con Gemini |
| `conf ≤ 0.50` | `needs_rescue` — Gemini intenta rescatar |

#### `_act_with_gemini(roberta_label, roberta_confidence, gemini_label)` — Acción
Combina los resultados de RoBERTa y Gemini con lógica escalonada:

**Zona media (conf 0.50 - 0.85):**
- Acuerdo RoBERTa = Gemini → `cross_validated` con boost de confianza (+0.05)
- Desacuerdo, conf > 0.65 → RoBERTa gana (tiene confianza razonable)
- Desacuerdo, conf ≤ 0.65 → Gemini gana (salvo labels opuestos → `ambiguous`)

**Zona baja (conf ≤ 0.50):**
- Acuerdo → `rescued` con boost de confianza
- Gemini dice "ambiguous" → `ambiguous`
- Labels opuestos (positive vs negative) → `ambiguous`
- Gemini da label claro → `rescued` (usa label de Gemini)

#### Prompt de Gemini
El prompt fue calibrado iterativamente con 20 textos de prueba (10 cross_validated + 10 ambiguos), logrando 90% de accuracy. Guías clave:
- Detección de sarcasmo sutil ("You mean..." = negativo)
- "Hoping for X" / "probably better" = neutral, no positivo
- Calls to action = neutral salvo con insultos
- En duda entre negativo y neutral → neutral

#### `_classify_with_gemini(texts)` — Clasificación en lote
Envía hasta 20 textos por llamada API. Parsea JSON con fallback regex. Retry automático con backoff exponencial (3 intentos).

#### `_record(results)` — Registro
Inserta en `sentiment_results` via `INSERT OR IGNORE` (idempotente).

#### `run(limit=1000, batch_size=64)` → `dict`
Ejecuta el ciclo en dos pasos: (1) RoBERTa en lotes de 64, (2) Gemini en lotes de 20 solo para textos que lo necesitan (~70% del total). Retorna métricas del ciclo.

---

## 8. Agente de Tendencias

**Archivo**: `src/agents/trends/trends_agent.py`

Detecta tendencias temáticas usando BERTopic para modelado y la métrica Δ para comparación temporal.

### Fórmula Δ

```
Δ = (w_current - mean_historical) / effective_std

donde:
  w_current         = peso del tópico en ventana actual (textos_tópico / total_textos_día)
  mean_historical   = media del peso diario en ventana histórica
  effective_std     = max(historical_std, STD_FLOOR=0.005)
```

**STD_FLOOR**: Valor mínimo para la desviación estándar. Evita que tópicos muy estables históricamente (σ ≈ 0) generen Δ artificialmente alto por división por valores cercanos a cero.

**Nota sobre el split temporal**: BERTopic se entrena con `fit_transform` sobre **todos** los textos (históricos + actuales combinados), y luego se separan las asignaciones por índice. No se usa `transform()` por separado porque HDBSCAN requiere `calculate_probabilities=True` para `transform()`, y esto genera un error KD-tree con corpus grandes. El split post-hoc es aceptable porque BERTopic es no supervisado y no tiene acceso a las etiquetas temporales durante el entrenamiento.

### Configuración
```python
DELTA_HIGH         = 1.5    # Umbral para tendencia/spike (calibrado con corpus real de 90 días)
DELTA_MODERATE     = 1.0    # Umbral para tendencia moderada
COVERAGE_THRESHOLD = 0.05   # 5% del corpus = emergente vs localizado
STD_FLOOR          = 0.005  # Piso mínimo de σ
MIN_TOPIC_TEXTS    = 10     # Mínimo textos en ventana actual

# Ventana adaptativa
CONFIDENCE_Z       = 1.96   # z para IC 95%
LIFECYCLE_ALPHA    = 2      # Multiplicador de vida media (α=2 → captura ~75% del ciclo)
FALLBACK_WINDOW_DAYS = 2    # Fallback si no se puede calcular half-life
MIN_WINDOW_HOURS   = 6      # Mínimo absoluto de ventana
MAX_WINDOW_DAYS    = 7      # Máximo absoluto de ventana
MIN_PEAK_TEXTS     = 20     # Mínimo de textos en pico para half-life
```

**Nota sobre thresholds**: Los valores 1.5/1.0 están calibrados para el corpus real de 90 días donde max Δ observado es 1.58 (tópico `oil_oil companies`, run `db7e2622`).

### Ventana adaptativa

La ventana de evaluación no es fija — se calcula automáticamente como:

```
W_eval = max(W_stat, W_lifecycle)
```

**W_stat = N_min / λ** — Garantiza estabilidad estadística. N_min se deriva del Teorema Central del Límite para proporciones binomiales (Cochran, 1977): `N_min = z² × p(1-p) / δ²` donde `p = COVERAGE_THRESHOLD`, `δ = p/2`. Con p=5%: N_min ≈ 292 textos.

**W_lifecycle = α × T_½** — Garantiza capturar el ciclo de vida del tópico. T_½ es la vida media empírica, medida ajustando decaimiento exponencial `f(t) = e^(-λt)` a la curva post-pico de cada tópico (Leskovec et al., 2009). α=2 captura ~75% del ciclo.

Con corpus denso (alto λ), W_lifecycle domina → ventana corta. Con corpus escaso (bajo λ), W_stat domina → ventana más larga.

**Resultado empírico** (corpus de 202K textos, 90 días):
- λ = 2,251 docs/día → W_stat = 0.13 días (3h)
- T_½ mediana = 18.9h → W_lifecycle = 1.57 días (37.8h)
- **W_eval = 1.57 días** (el constraint de ciclo de vida domina)

### Stopwords de Reddit
Se añade una lista `REDDIT_STOPWORDS` al `CountVectorizer` de BERTopic para filtrar meta-conversación:
```python
REDDIT_STOPWORDS = [
    "comment", "comments", "post", "posts", "response", "responses",
    "said", "says", "saying", "explained", "replied", "reply",
    "upvote", "downvote", "edit", "deleted", "removed",
    "thread", "subreddit", "reddit", "mod", "moderator", ...
]
```
Sin esta lista, BERTopic genera tópicos de meta-conversación (ej: `response_said_explained_comments`) que no tienen contenido político.

### Ciclo ReAct

#### `_observe(limit)` — Observación
Carga textos con timestamps. Calcula la ventana adaptativa (o usa la forzada) y retorna `(historical_texts, current_texts)` separados por el cutoff `max_ts - W_eval`.

#### `_reason(historical_texts, current_texts)` — Razonamiento
1. Combina históricos + actuales en `all_texts`
2. `fit_transform(all_docs)` — entrena BERTopic sobre todos los textos
3. Separa `all_topic_ids` por índice: `hist_topic_ids = all_topic_ids[:len(historical_texts)]`
4. Calcula `_calculate_temporal_stats()` con los cuatro arrays separados
5. Retorna `hist_topic_ids, curr_topic_ids, temporal_stats`

#### `_act(topic_id, stats)` — Acción
Aplica la lógica de decisión:

| Condición | Decisión |
|-----------|----------|
| Δ ≥ 1.5 y coverage > 5% | `emerging_trend` — tendencia emergente |
| Δ ≥ 1.5 y coverage ≤ 5% | `localized_spike` — spike localizado, monitorear |
| 1.0 ≤ Δ < 1.5 y peso actual > media | `moderate_trend` — tendencia moderada |
| 1.0 ≤ Δ < 1.5 y peso actual ≤ media | `discarded` — pico pasajero |
| Δ < 1.0 | `discarded` — no es tendencia |

#### `_record(...)` — Registro
Guarda asignaciones de tópico en `topic_assignments` y resultados en `trend_analysis`. El campo `daily_weights_json` almacena la evolución temporal completa de cada tópico.

#### `run(limit=50000)` → `dict`
Ejecuta el ciclo completo. Retorna métricas incluyendo top 15 tópicos por Δ.

---

## 9. Agente de Validación

**Archivo**: `src/agents/validation/validation_agent.py`

Sintetiza los resultados de sentimiento y tendencias para generar un reporte final con alertas y contexto político.

### Funcionalidades

1. **Sistema de alertas**: Clasifica tendencias en alertas críticas (Δ ≥ 3.0 y negatividad ≥ 70%) o informativas (Δ ≥ 2.0).
2. **Contexto político (LLM)**: Usa Gemini Flash para generar un resumen ejecutivo y contexto político por cada tendencia, basándose en textos representativos del tópico.
3. **Generación de reportes**: Produce un reporte .md con gráficos (distribución de sentimiento, confianza, Δ por tópico, word clouds).

### Umbrales configurables
```python
ALERT_CRITICAL_DELTA   = 3.0    # Δ mínimo para alerta crítica
ALERT_CRITICAL_NEG_PCT = 0.70   # Negatividad mínima para alerta crítica
ALERT_INFORMATIVE_DELTA = 2.0   # Δ mínimo para alerta informativa
```

---

## 10. Orquestador LangGraph

**Archivo**: `src/orchestrator/orchestrator.py`

Define un `StateGraph` de LangGraph que coordina la ejecución de todos los agentes con **decisiones condicionales**:

### Flujo del grafo
```
preprocess → sentiment → trends → [validation | no_trends_report] → finalize
```

### Decisiones condicionales (valor agentic)

| Nodo | Decisión | Alternativa |
|------|----------|-------------|
| `should_run_sentiment` | ¿Hay textos pendientes? | Sí → sentiment, No → trends |
| `should_run_trends` | ¿Hay resultados de sentimiento? | Sí → trends, No → finalize |
| `should_run_validation` | ¿Hay tendencias relevantes? | Sí → validation, No → no_trends_report |

La decisión más importante es `should_run_validation`: si el agente de tendencias descartó **todos** los tópicos, el orquestador genera un reporte de ausencia y no invoca al agente de validación (ahorrando recursos y evitando reportes con ruido).

### Estado compartido
```python
class OrchestratorState(TypedDict, total=False):
    run_id: str
    db_path: str
    preprocess_result: dict
    sentiment_result: dict
    trends_result: dict
    validation_result: dict
    steps_completed: list[str]
```

Cada decisión y resultado se persiste en la BD (`orchestration_runs`), permitiendo auditar el camino tomado.

---

## 11. Pipeline Tradicional (Baseline)

**Archivo**: `src/pipeline/traditional_pipeline.py`

Implementa un pipeline secuencial sin decisiones agentic para comparación:
- Clasifica con RoBERTa argmax directo (sin umbrales, sin Gemini, sin abstención)
- Ejecuta BERTopic sin filtrado por Δ
- Reporta todos los tópicos sin evaluar relevancia estadística

Sirve como baseline para demostrar el valor del enfoque agentic.

---

## 12. Evaluación del Protocolo Experimental

**Archivo**: `scripts/run_evaluation.py`

Implementa todas las métricas del protocolo experimental requeridas. Cada sección es independiente y se puede correr por separado.

### `--sentiment` — Métricas de sentimiento

1. **Distribución de confianza**: Histograma de `roberta_confidence` en 3 rangos (alta/media/baja)
2. **Tasa de ambigüedad**: % de textos con `decision = 'ambiguous'` (objetivo: < 10%)
3. **Acuerdo inter-modelo RoBERTa vs Gemini**: De los textos `cross_validated` y `rescued`, qué % coincidieron ambos modelos.

### `--topics` — Coherencia temática

Calcula **c_v** y **UMass** usando gensim sobre los tópicos del último run de BERTopic.
- **c_v**: Co-ocurrencia de palabras en contexto. Rango 0-1, ideal > 0.55.
- **UMass**: Co-ocurrencia en corpus. Rango -∞ a 0, ideal > -2.0.

**Nota técnica UMass**: Las palabras clave de BERTopic se filtran para incluir solo aquellas presentes en el diccionario gensim antes de calcular UMass. Sin este filtro, palabras ausentes del corpus generan `log(0/0) = nan`.

### Resultados del protocolo experimental (run Abril 2026, 203,275 textos)

**Sentimiento:**
- negative: 69.4%, neutral: 26.4%, positive: 3.2%, ambiguous: 1.0%
- Confianza promedio: 0.7627; confianza mediana: 0.7958; confianza alta (>0.85): 30.2%
- Tasa de ambigüedad: 0.98% ✅ (objetivo < 10%)
- Cross-validación con Gemini 2.5 Flash Lite para zona de incertidumbre
- Decisiones: accepted 30.2%, cross_validated 64.2%, rescued 4.6%, ambiguous 1.0%

**Ground truth (DeepSeek V3 pseudo-labels, validación manual κ=0.9651):**
- Accuracy agentic: 0.7510
- F1-macro: 0.599
- Δ accuracy (agentic - pipeline): +0.08

**Tópicos (379 tópicos, ventana adaptativa W=1.57 días):**
- c_v = 0.776 ✅ (objetivo > 0.55)
- Jaccard stability (3 runs) = 0.731 ✅ (objetivo > 0.70)
- Tendencias relevantes: 7 de 379 tópicos (0 alertas críticas, 2 informativas)
- Top tópico por Δ: `125_fetterman_john fetterman_lamb_stroke` Δ=6.44 (localized_spike, INFORMATIVE)

> **Nota UMass**: El score bajo es esperado en corpora de redes sociales. UMass fue calibrado sobre textos formales (Wikipedia, noticias); en Reddit el léxico es más diverso y los términos no co-ocurren tan densamente. c_v (que usa co-ocurrencia en ventana deslizante) es más robusto para este tipo de datos y el valor 0.776 es excelente.

**Estabilidad (Jaccard 3 runs, 5000 textos):**
- **Jaccard promedio: 0.731 ✅** (objetivo > 0.70)

**Ventana adaptativa:**
- λ = 2,251 docs/día → W_stat = 0.13 días (3h)
- T_½ mediana = 18.9h → W_lifecycle = 1.57 días (37.8h)
- **W_eval = 1.57 días** (el constraint de ciclo de vida domina)

### `--stability` — Estabilidad de clustering

Ejecuta BERTopic 3 veces con los mismos datos (reutilizando embeddings para eficiencia) y calcula **Jaccard similarity** entre los tópicos resultantes.

```
Jaccard = palabras en común / total palabras distintas entre dos tópicos
```

Para cada par de runs, encuentra el mejor match por Jaccard para cada tópico. Promedia todos los scores. Ideal > 0.70.

### `--groundtruth` — Métricas contra pseudo ground truth DeepSeek V3

Calcula Accuracy, Precision, Recall y F1 macro comparando las predicciones de RoBERTa contra las pseudo-etiquetas generadas por DeepSeek V3. Excluye textos marcados como `ambiguous` por el agente (no hay predicción de clase). Muestra:
- Reporte por clase con precision/recall/F1 y support
- Matriz de confusión (filas=true, columnas=pred)
- Agreement rate global

### `--manual` — Validación manual del ground truth

Compara etiquetas manuales (CSV anotado por humano) contra las pseudo-etiquetas de DeepSeek V3 para auditar la calidad del pseudo-labeling. Requiere un CSV con columna `manual_label` completada. Reporta:
- Distribución de etiquetas (manual vs DeepSeek)
- Accuracy de acuerdo y reporte por clase
- Tipos de error específicos (manual → DeepSeek)

### `--compare` — Comparación agentic vs pipeline tradicional

Compara ambos enfoques sobre el mismo ground truth:
- **Pipeline**: `roberta_label` directo sobre todos los textos (sin umbrales, sin Gemini)
- **Agentic**: `final_label` con mecanismo de cuatro caminos (accepted/cross_validated/rescued/ambiguous)

Reporta accuracy y F1 macro de ambos, y analiza la **ganancia por abstención informada**: evalúa qué accuracy obtiene el pipeline sobre los textos que el agente clasifica como `ambiguous`, demostrando que forzar una etiqueta en esos casos introduce error cercano al azar.

### `--delta` — Sensibilidad de parámetros Δ

Análisis de sensibilidad del agente de tendencias sobre distintos valores de `DELTA_HIGH` y `DELTA_MODERATE`. Incluye:
- Distribución de valores Δ (min, max, media, mediana)
- Tabla de sensibilidad: cuántos tópicos se detectan por cada threshold
- Separación emerging (coverage > 5%) vs localized (coverage ≤ 5%)

### `--failure-modes` — Análisis estructurado de failure modes

Análisis cualitativo y cuantitativo de **por qué** falla el sistema, no solo cuánto falla. Incluye 8 sub-análisis:

1. **FM1 — Patrones de confusión**: Qué pares de clases se confunden más (ej. neutral→negative) y con qué frecuencia
2. **FM2 — Errores por tipo de decisión**: Tasa de error en textos `accepted` vs `cross_validated`, confirmando que la zona de confianza intermedia es más propensa a error
3. **FM3 — Errores por rango de confianza**: Desglose de tasa de error por banda de confianza (alta/media-alta/media-baja)
4. **FM4 — Errores por longitud de texto**: Relación entre longitud del texto y probabilidad de clasificación incorrecta
5. **FM5 — Sarcasmo e ironía**: Detección de indicadores de sarcasmo (`/s`, `lol`, comillas irónicas, adverbios irónicos, puntuación enfática) y su prevalencia relativa en errores vs aciertos (ratio)
6. **FM6 — Comportamiento de Gemini en errores cross_validated/rescued**: Cuántas veces Gemini habría dado la respuesta correcta
7. **FM7 — Ejemplos representativos**: Textos concretos por cada tipo de confusión dominante, con confianza, decisión y razonamiento de DeepSeek
8. **FM8 — Análisis de abstención**: Distribución real de los textos `ambiguous` y accuracy hipotética si se hubieran clasificado

```bash
python -m scripts.run_evaluation --failure-modes
```

### `--latency` — Latencia comparativa

Mide y compara:
1. **RoBERTa directo** — sin lógica agentic
2. **Agente ReAct** — con razonamiento y validación Gemini
3. **BERTopic directo** — sin agente de tendencias

Reporta tiempo total y ms/texto para cada componente, y el overhead del agente ReAct respecto al pipeline directo.

---

## 13. Interfaz Streamlit

**Archivo**: `app.py`

Interfaz web con 4 páginas para visualizar resultados:

1. **Reporte**: Muestra el reporte más reciente generado por el agente de validación (resumen ejecutivo, alertas, tendencias con word clouds y contexto político).
2. **Metricas Agentic vs Pipeline**: 9 secciones comparativas — accuracy, calibración por tier, predicción selectiva, errores evitados, filtrado de tendencias, interpretabilidad, ground truth, ejecución condicional.
3. **Explorador de Datos**: Distribución de sentimiento, muestras de texto por decisión, tendencias detectadas, historial de ejecuciones del orquestador.
4. **Arquitectura**: Diagrama del flujo del sistema y tabla de decisiones por componente.

```bash
streamlit run app.py
```

---

## 14. Scripts de Ejecución

### Recolección

```bash
# Recolección histórica 90 días — Arctic Shift (RECOMENDADO para corpus completo)
python -m scripts.collect_data --arctic --days 90

# Recolección tiempo real — PRAW (últimos ~7 días)
python -m scripts.collect_data --days 7

# Recolección continua — PRAW en bucle
python -m scripts.collect_data --continuous --interval 3600

# Demo rápida: recolecta + preprocesa últimos N minutos
python -m scripts.collect_data --live --minutes 30
```

### Preprocesamiento

```bash
# Procesar todos los textos pendientes
python -m scripts.preprocess_data

# Solo ver estadísticas
python -m scripts.preprocess_data --stats
```

### Sentimiento

```bash
# Analizar todos los textos pendientes
python -m scripts.run_sentiment

# Con límite (prueba rápida)
python -m scripts.run_sentiment --limit 500

# Ajustar umbrales
python -m scripts.run_sentiment --high-conf 0.90 --low-conf 0.55

# Ver estadísticas sin analizar
python -m scripts.run_sentiment --stats

# Verificar clasificaciones visualmente (100 comentarios aleatorios)
python -m scripts.test_sentiment
```

### Tendencias

```bash
# Detectar tendencias (corre BERTopic completo)
python -m scripts.run_trends

# Con límite de textos (recomendado: 200000 para corpus completo)
python -m scripts.run_trends --limit 200000

# Forzar número de tópicos
python -m scripts.run_trends --n-topics 30

# Ajustar ventanas temporales
python -m scripts.run_trends --historical-days 60 --current-days 7

# Ver resultados del último run sin reejecutar
python -m scripts.run_trends --results

# Calcular coherencia c_v y UMass
python -m scripts.run_trends --coherence
```

### Inspección visual

```bash
# Ver clasificaciones de sentimiento (texto + label)
python -m scripts.inspect_sentiment --n 30

# Ver tópicos de tendencias
python -m scripts.inspect_trends

# Filtrar por decisión
python -m scripts.inspect_trends --decision emerging_trend --n 10

# Ver textos de un tópico específico
python -m scripts.inspect_trends --topic 5
```

### Orquestador

```bash
# Ejecución completa del orquestador LangGraph
python -m scripts.run_orchestrator
```

### Pipeline Tradicional (Baseline)

```bash
# Pipeline completo (sentimiento + tendencias)
python -m scripts.run_pipeline

# Solo sentimiento con límite
python -m scripts.run_pipeline --limit-sentiment 5000

# Solo tendencias
python -m scripts.run_pipeline --trends-only

# Ajustar ventana actual
python -m scripts.run_pipeline --current-days 2
```

### Validación

```bash
# Ejecutar agente de validación sobre último run de tendencias
python -m scripts.run_validation
```

### Utilidades

```bash
# Re-clasificar textos con Gemini (one-shot, ya ejecutado sobre 202K textos)
python -m scripts.reclassify_with_gemini

# Análisis de half-life y ventana adaptativa
python -m scripts.window_analysis

# Exportar muestra aleatoria para etiquetado manual
python -m scripts.export_manual_sample
```

### Evaluación experimental

```bash
# Todas las métricas
python -m scripts.run_evaluation --all

# Solo sentimiento (distribución, ambigüedad, acuerdo inter-modelo)
python -m scripts.run_evaluation --sentiment

# Métricas contra ground truth DeepSeek V3 (accuracy, F1, confusion matrix)
python -m scripts.run_evaluation --groundtruth

# Validación manual del ground truth (300 textos anotados)
python -m scripts.run_evaluation --manual
python -m scripts.run_evaluation --manual --manual-csv ruta/al/archivo.csv

# Comparación agentic vs pipeline tradicional (mismas métricas, mismo GT)
python -m scripts.run_evaluation --compare

# Sensibilidad de parámetros Δ del agente de tendencias
python -m scripts.run_evaluation --delta

# Análisis de failure modes (por qué falla el sistema)
python -m scripts.run_evaluation --failure-modes

# Solo coherencia temática (c_v, UMass)
python -m scripts.run_evaluation --topics

# Estabilidad de clustering (3 runs BERTopic — tarda ~30 min)
python -m scripts.run_evaluation --stability --stability-limit 5000

# Latencia comparativa (muestra de 200 textos)
python -m scripts.run_evaluation --latency --latency-sample 200
```

---

## 15. Flujo Completo de Datos

```
╔══════════════════════════════════════════════════════════╗
║           RECOLECCIÓN                                     ║
║                                                          ║
║  Arctic Shift API ──► 90 días históricos uniformes       ║
║  PRAW --continuous ──► actualizaciones diarias           ║
╚══════════════════╦═══════════════════════════════════════╝
                   │
                   ▼
        SQLite: posts, comments
                   │
                   ▼
╔══════════════════════════════════════════════════════════╗
║           PREPROCESAMIENTO                               ║
║                                                          ║
║  Filtra: bots, deleted, vacíos, < 10 palabras           ║
║  Genera: text_for_sentiment  → RoBERTa                  ║
║          text_for_topics     → BERTopic                 ║
╚══════════════════╦═══════════════════════════════════════╝
                   │
                   ▼
        SQLite: preprocessed_texts
                   │
          ┌────────┴────────┐
          ▼                 ▼
╔══════════════╗   ╔══════════════════════╗
║  AGENTE      ║   ║  AGENTE TENDENCIAS   ║
║  SENTIMIENTO ║   ║                      ║
║              ║   ║  BERTopic detecta    ║
║  RoBERTa     ║   ║  tópicos             ║
║  + Gemini    ║   ║                      ║
║  ReAct:      ║   ║  Δ = (curr - mean)   ║
║  accepted /  ║   ║      / effective_std ║
║  cross_val / ║   ║                      ║
║  rescued /   ║   ║  emerging_trend /    ║
║  ambiguous   ║   ║                      ║
║              ║   ║  localized_spike /   ║
║              ║   ║  moderate_trend /    ║
║              ║   ║  discarded           ║
╚══════╦═══════╝   ╚══════════╦═══════════╝
       │                      │
       ▼                      ▼
 sentiment_results    topic_assignments
                      trend_analysis
       │                      │
       └──────────┬───────────┘
                  ▼
╔══════════════════════════════════════════════════════════╗
║  ORQUESTADOR LANGGRAPH                                    ║
║  ¿Hay tendencias relevantes?                             ║
║  SÍ → Agente de Validación (alertas + LLM + reporte)    ║
║  NO → Reporte de ausencia (ahorra recursos)              ║
╚══════════════════════════════════════════════════════════╝
                  │
                  ▼
╔══════════════════════════════════════════════════════════╗
║  AGENTE DE VALIDACIÓN                                     ║
║  Alertas (critical/informative)                          ║
║  Contexto político (Gemini Flash) + reporte final        ║
╚══════════════════════════════════════════════════════════╝
```

### Ejemplo de transformación de texto

**Comentario original:**
```
Check out [this analysis](https://example.com)!!!

I think **Trump's** policy on NATO is absolutely TERRIBLE. The u/someuser
pointed this out. This is going to hurt us badly 😡😡😡
```

**`text_for_sentiment`:**
```
Check out this analysis! I think Trump's policy on NATO is absolutely TERRIBLE.
@user pointed this out. This is going to hurt us badly http
```
→ Caso preservado, URLs → `http`, menciones → `@user`, sin emojis.

**`text_for_topics`:**
```
Check out this analysis! I think Trump's policy on NATO is absolutely TERRIBLE.
pointed this out. This is going to hurt us badly
```
→ URLs y menciones eliminadas, caso preservado (Trump, NATO aportan semántica).

---

## 16. Parámetros Configurables

| Parámetro | Dónde | Valor | Efecto |
|-----------|-------|-------|--------|
| `TARGET_SUBREDDITS` | settings.py | `["politics"]` | Agrega/quita subreddits |
| `COMMENTS_PER_POST` | arctic_collector.py | 200 | Comentarios por post vía Arctic Shift |
| `REQUEST_SLEEP` | arctic_collector.py | 0.5s | Pausa entre requests a Arctic Shift |
| `RATE_LIMIT_SLEEP` | settings.py | 1s | Pausa entre requests a PRAW |
| `MIN_WORD_COUNT` | settings.py | 10 | Filtro de calidad mínimo |
| `BERTOPIC_MIN_WORDS` | settings.py | 15 | BERTopic necesita más contexto |
| `HIGH_CONF_THRESHOLD` | sentiment_agent.py | 0.85 | Umbral para aceptar directo |
| `LOW_CONF_THRESHOLD` | sentiment_agent.py | 0.50 | Umbral para rescate con Gemini |
| `MID_CONF_THRESHOLD` | sentiment_agent.py | 0.65 | Desempate en zona media |
| `GEMINI_MODEL` | sentiment_agent.py | gemini-2.5-flash-lite | Modelo LLM para cross-validación |
| `GEMINI_BATCH_SIZE` | sentiment_agent.py | 20 | Textos por llamada API a Gemini |
| `ALERT_CRITICAL_DELTA` | validation_agent.py | 3.0 | Δ mínimo para alerta crítica |
| `ALERT_INFORMATIVE_DELTA` | validation_agent.py | 2.0 | Δ mínimo para alerta informativa |
| `DELTA_HIGH` | trends_agent.py | 1.5 | Umbral Δ para tendencia/spike (calibrado corpus real) |
| `DELTA_MODERATE` | trends_agent.py | 1.0 | Umbral Δ para tendencia moderada |
| `COVERAGE_THRESHOLD` | trends_agent.py | 0.05 | 5% cobertura = emergente vs localizado |
| `STD_FLOOR` | trends_agent.py | 0.005 | Piso mínimo de σ para estabilidad numérica |
| `LIFECYCLE_ALPHA` | trends_agent.py | 2 | Multiplicador de vida media para ventana adaptativa (α=2 → 75% del ciclo) |
| `FALLBACK_WINDOW_DAYS` | trends_agent.py | 2 | Ventana fallback si no se puede calcular half-life |
| `MIN_WINDOW_HOURS` | trends_agent.py | 6 | Mínimo absoluto de ventana de evaluación |
| `MAX_WINDOW_DAYS` | trends_agent.py | 7 | Máximo absoluto de ventana de evaluación |
| `min_topic_size` | trends_agent.py | 50 | Tamaño mínimo de tópico BERTopic (~365 tópicos con corpus de 90 días) |
| `MIN_TOPIC_TEXTS` | trends_agent.py | 10 | Mínimo textos en ventana actual para evaluar tópico |
