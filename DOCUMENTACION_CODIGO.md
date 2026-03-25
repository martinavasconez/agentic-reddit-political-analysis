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
9. [Evaluación del Protocolo Experimental](#9-evaluacion)
10. [Scripts de Ejecución](#10-scripts-de-ejecucion)
11. [Flujo Completo de Datos](#11-flujo-completo)
12. [Parámetros Configurables](#12-parametros-configurables)

---

## 1. Visión General

Este proyecto implementa una **arquitectura agentic** para el análisis de sentimiento y detección de tendencias en el discurso político de Reddit. Los agentes siguen el patrón **ReAct** (Reason + Act): observan datos, razonan sobre ellos y toman decisiones basadas en umbrales.

### Componentes implementados

| Componente | Descripción | Estado |
|-----------|-------------|--------|
| Recolección histórica | Arctic Shift API — 90 días de datos históricos uniformes | ✅ |
| Recolección en tiempo real | PRAW — posts más recientes | ✅ |
| Preprocesamiento | Limpieza y normalización para RoBERTa y BERTopic | ✅ |
| Agente de Sentimiento | RoBERTa + VADER con patrón ReAct | ✅ |
| Agente de Tendencias | BERTopic + cálculo de Δ temporal | ✅ |
| Evaluación experimental | c_v, UMass, Jaccard, latencia comparativa | ✅ |
| Agente de Validación | Síntesis de sentimiento + tendencias | 🔄 Pendiente |
| Orquestador LangGraph | Coordinación del pipeline completo | 🔄 Pendiente |

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
│   └── agents/
│       ├── sentiment/
│       │   └── sentiment_agent.py  # Agente ReAct de sentimiento
│       └── trends/
│           └── trends_agent.py     # Agente ReAct de tendencias
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
| `decision` | TEXT | Decisión ReAct: `accepted/cross_validated/ambiguous` |
| `final_label` | TEXT | Etiqueta final: `positive/negative/neutral/ambiguous` |
| `final_confidence` | REAL | Confianza final (puede tener boost de VADER) |
| `vader_compound` | REAL | Score VADER (-1 a 1), solo en `cross_validated` |
| `vader_label` | TEXT | Etiqueta VADER, solo en `cross_validated` |
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
| `consecutive_growth_days` | INTEGER | Días consecutivos creciendo |
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

#### `collect_historical(subreddit_name, days, max_comments_per_post)`

**NOTA**: Este método usa búsqueda Lucene de Reddit (`timestamp:EPOCH1..EPOCH2`) que **ya no funciona** con la API actual de Reddit. Está mantenido en el código pero no se usa. Para recolección histórica usar `ArcticCollector`.

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
- **VADER**: `vaderSentiment` — léxico de sentimiento basado en reglas. Rápido pero falla con sarcasmo político.

### Umbrales configurables
```python
HIGH_CONF_THRESHOLD = 0.85   # Por encima: acepta directamente
LOW_CONF_THRESHOLD  = 0.50   # Por debajo: marca como ambiguo
```

### Ciclo ReAct

#### `_observe(limit)` — Observación
Consulta `get_unanalyzed_texts_for_sentiment()` — LEFT JOIN para obtener textos sin análisis previo. Garantiza idempotencia.

#### `_reason(roberta_scores)` — Razonamiento
Analiza el output de RoBERTa y decide:

| Condición | Decisión |
|-----------|----------|
| `conf > 0.85` | `accepted` — RoBERTa es suficientemente seguro |
| `0.50 < conf ≤ 0.85` | `cross_validated` — necesita validación con VADER |
| `conf ≤ 0.50` | `ambiguous` — demasiada incertidumbre |

#### `_act(text, decision, roberta_label, roberta_confidence)` — Acción
Ejecuta la acción según la decisión:

- **`accepted`**: Usa directamente el resultado de RoBERTa.
- **`cross_validated`**: Llama a VADER y compara:
  - Si coinciden con RoBERTa → boost de confianza: `final_confidence = min(conf + 0.05, 1.0)`
  - Si discrepan → **RoBERTa gana** (mejor comprensión de contexto político). VADER queda registrado como evidencia.
- **`ambiguous`**: `final_label = "ambiguous"`, se excluye del análisis agregado.

**Decisión de diseño clave**: VADER no tiene poder de veto. Solo puede aumentar la confianza cuando coincide. Esto redujo la tasa de ambiguos del 38.2% inicial al 3.2%.

#### `_record(results)` — Registro
Inserta en `sentiment_results` via `INSERT OR IGNORE` (idempotente).

#### `run(limit=1000, batch_size=64)` → `dict`
Ejecuta el ciclo completo. Procesa en lotes de 64 textos para eficiencia con RoBERTa. Retorna métricas del ciclo.

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
DELTA_HIGH         = 1.5    # Umbral para tendencia/spike (calibrado con corpus real de 89 días)
DELTA_MODERATE     = 1.0    # Umbral para tendencia moderada
COVERAGE_THRESHOLD = 0.05   # 5% del corpus = emergente vs localizado
STD_FLOOR          = 0.005  # Piso mínimo de σ
MIN_TOPIC_TEXTS    = 10     # Mínimo textos en ventana actual
HISTORICAL_DAYS    = 60     # Días de ventana histórica
CURRENT_DAYS       = 7      # Días de ventana de evaluación
```

**Nota sobre thresholds**: Los valores 1.5/1.0 están calibrados para el corpus real de 89 días donde max Δ observado es ~3.85.

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
Carga textos con timestamps. Retorna `(historical_texts, current_texts)` separados por el cutoff `max_ts - CURRENT_DAYS`.

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
| 1.0 ≤ Δ < 1.5 y 3+ días creciendo | `moderate_trend` — tendencia moderada |
| 1.0 ≤ Δ < 1.5 y decreciendo | `discarded` — pico pasajero |
| Δ < 1.0 | `discarded` — no es tendencia |

#### `_record(...)` — Registro
Guarda asignaciones de tópico en `topic_assignments` y resultados en `trend_analysis`. El campo `daily_weights_json` almacena la evolución temporal completa de cada tópico.

#### `run(limit=50000)` → `dict`
Ejecuta el ciclo completo. Retorna métricas incluyendo top 15 tópicos por Δ.

---

## 9. Evaluación del Protocolo Experimental

**Archivo**: `scripts/run_evaluation.py`

Implementa todas las métricas del protocolo experimental requeridas. Cada sección es independiente y se puede correr por separado.

### `--sentiment` — Métricas de sentimiento

1. **Distribución de confianza**: Histograma de `roberta_confidence` en 3 rangos (alta/media/baja)
2. **Tasa de ambigüedad**: % de textos con `decision = 'ambiguous'` (objetivo: < 10%)
3. **Acuerdo inter-modelo RoBERTa vs VADER**: De los textos `cross_validated`, qué % coincidieron ambos modelos. Un acuerdo > 70% valida la consistencia de RoBERTa sin necesidad de ground truth.

### `--topics` — Coherencia temática

Calcula **c_v** y **UMass** usando gensim sobre los tópicos del último run de BERTopic.
- **c_v**: Co-ocurrencia de palabras en contexto. Rango 0-1, ideal > 0.55.
- **UMass**: Co-ocurrencia en corpus. Rango -∞ a 0, ideal > -2.0.

**Nota técnica UMass**: Las palabras clave de BERTopic se filtran para incluir solo aquellas presentes en el diccionario gensim antes de calcular UMass. Sin este filtro, palabras ausentes del corpus generan `log(0/0) = nan`.

### Resultados del protocolo experimental (run Marzo 2026, 201,568 textos)

**Sentimiento:**
- negative: 69.3%, neutral: 21.9%, positive: 3.8%, ambiguous: 5.0%
- Confianza promedio: 0.742; confianza alta (>0.85): 30.2%
- Tasa de ambigüedad: 4.98% ✅ (objetivo < 10%)
- Acuerdo inter-modelo RoBERTa-VADER: 42.9% (zona genuinamente ambigua)

**Tópicos (run `db7e2622`, 236 tópicos):**
- c_v = 0.776 ✅ (objetivo > 0.55)
- Jaccard stability (3 runs) = 0.794 ✅ (objetivo > 0.70)
- Top tópico: `trans_women_sports` Δ=+3.85 (🔥 EMERGING TREND, cobertura 5.4%)

> **Nota UMass**: El score bajo es esperado en corpora de redes sociales. UMass fue calibrado sobre textos formales (Wikipedia, noticias); en Reddit el léxico es más diverso y los términos no co-ocurren tan densamente. c_v (que usa co-ocurrencia en ventana deslizante) es más robusto para este tipo de datos y el valor 0.77 es excelente.

**Estabilidad (Jaccard 3 runs, 5000 textos):**
- Run 1 vs 2: 0.7940 / Run 1 vs 3: 0.7828 / Run 2 vs 3: 0.8037
- **Jaccard promedio: 0.7935 ✅** (objetivo > 0.70)

### `--stability` — Estabilidad de clustering

Ejecuta BERTopic 3 veces con los mismos datos (reutilizando embeddings para eficiencia) y calcula **Jaccard similarity** entre los tópicos resultantes.

```
Jaccard = palabras en común / total palabras distintas entre dos tópicos
```

Para cada par de runs, encuentra el mejor match por Jaccard para cada tópico. Promedia todos los scores. Ideal > 0.70.

### `--latency` — Latencia comparativa

Mide y compara:
1. **RoBERTa directo** — sin lógica agentic
2. **Agente ReAct** — con razonamiento y validación VADER
3. **BERTopic directo** — sin agente de tendencias

Reporta tiempo total y ms/texto para cada componente, y el overhead del agente ReAct respecto al pipeline directo.

---

## 10. Scripts de Ejecución

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

### Evaluación experimental

```bash
# Todas las métricas
python -m scripts.run_evaluation --all

# Solo sentimiento (distribución, ambigüedad, acuerdo inter-modelo)
python -m scripts.run_evaluation --sentiment

# Solo coherencia temática (c_v, UMass)
python -m scripts.run_evaluation --topics

# Estabilidad de clustering (3 runs BERTopic — tarda ~30 min)
python -m scripts.run_evaluation --stability --stability-limit 5000

# Latencia comparativa (muestra de 200 textos)
python -m scripts.run_evaluation --latency --latency-sample 200
```

---

## 11. Flujo Completo de Datos

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
║  + VADER     ║   ║                      ║
║  ReAct:      ║   ║  Δ = (curr - mean)   ║
║  accepted /  ║   ║      / effective_std ║
║  cross_val / ║   ║                      ║
║  ambiguous   ║   ║  emerging_trend /    ║
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
║  EVALUACIÓN EXPERIMENTAL                                  ║
║                                                          ║
║  --sentiment: confianza, ambigüedad, acuerdo VADER       ║
║  --topics:    c_v, UMass                                 ║
║  --stability: Jaccard similarity entre 3 runs            ║
║  --latency:   comparativa con/sin agente                 ║
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

## 12. Parámetros Configurables

| Parámetro | Dónde | Valor | Efecto |
|-----------|-------|-------|--------|
| `TARGET_SUBREDDITS` | settings.py | `["politics"]` | Agrega/quita subreddits |
| `COMMENTS_PER_POST` | arctic_collector.py | 200 | Comentarios por post vía Arctic Shift |
| `REQUEST_SLEEP` | arctic_collector.py | 0.5s | Pausa entre requests a Arctic Shift |
| `RATE_LIMIT_SLEEP` | settings.py | 1s | Pausa entre requests a PRAW |
| `MIN_WORD_COUNT` | settings.py | 10 | Filtro de calidad mínimo |
| `BERTOPIC_MIN_WORDS` | settings.py | 15 | BERTopic necesita más contexto |
| `HIGH_CONF_THRESHOLD` | sentiment_agent.py | 0.85 | Umbral para aceptar directo |
| `LOW_CONF_THRESHOLD` | sentiment_agent.py | 0.50 | Umbral para marcar ambiguo |
| `DELTA_HIGH` | trends_agent.py | 1.5 | Umbral Δ para tendencia/spike (calibrado corpus real) |
| `DELTA_MODERATE` | trends_agent.py | 1.0 | Umbral Δ para tendencia moderada |
| `COVERAGE_THRESHOLD` | trends_agent.py | 0.05 | 5% cobertura = emergente vs localizado |
| `STD_FLOOR` | trends_agent.py | 0.005 | Piso mínimo de σ para estabilidad numérica |
| `HISTORICAL_DAYS` | trends_agent.py | 60 | Días de ventana histórica para baseline |
| `CURRENT_DAYS` | trends_agent.py | 7 | Días de ventana actual para Δ |
| `min_topic_size` | trends_agent.py | 50 | Tamaño mínimo de tópico BERTopic (~236 tópicos con corpus de 89 días) |
| `MIN_TOPIC_TEXTS` | trends_agent.py | 10 | Mínimo textos en ventana actual para evaluar tópico |
