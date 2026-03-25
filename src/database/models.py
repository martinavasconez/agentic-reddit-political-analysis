"""
Esquema de la base de datos SQLite.
Define las tablas para posts, comentarios y texto preprocesado.
"""

SCHEMA_SQL = """
-- Posts de Reddit
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,              -- Reddit post ID (e.g., "1abc23")
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT DEFAULT '',
    author TEXT,
    score INTEGER DEFAULT 0,
    upvote_ratio REAL DEFAULT 0.0,
    num_comments INTEGER DEFAULT 0,
    created_utc REAL NOT NULL,
    url TEXT,
    is_self INTEGER DEFAULT 0,        -- Boolean: 1 if self post, 0 if link
    permalink TEXT,
    collected_at TEXT NOT NULL,        -- Timestamp de cuándo se recolectó
    UNIQUE(id)
);

-- Comentarios de Reddit
CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,              -- Reddit comment ID
    post_id TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    body TEXT NOT NULL,
    author TEXT,
    score INTEGER DEFAULT 0,
    created_utc REAL NOT NULL,
    parent_id TEXT,                   -- ID del padre (post o comentario)
    is_root INTEGER DEFAULT 0,       -- 1 si es comentario de nivel superior
    depth INTEGER DEFAULT 0,
    controversiality INTEGER DEFAULT 0,
    collected_at TEXT NOT NULL,
    UNIQUE(id),
    FOREIGN KEY (post_id) REFERENCES posts(id)
);

-- Texto preprocesado (una fila por comentario/post procesado)
CREATE TABLE IF NOT EXISTS preprocessed_texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,          -- ID del post o comentario original
    source_type TEXT NOT NULL,        -- 'post' o 'comment'
    subreddit TEXT NOT NULL,
    original_text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,       -- Texto limpio base
    text_for_sentiment TEXT NOT NULL, -- Texto optimizado para RoBERTa (más corto)
    text_for_topics TEXT NOT NULL,    -- Texto optimizado para BERTopic (más contexto)
    word_count INTEGER NOT NULL,
    created_utc REAL NOT NULL,        -- Fecha del contenido original
    processed_at TEXT NOT NULL,
    is_valid INTEGER DEFAULT 1,       -- 1 si pasa filtros de calidad
    UNIQUE(source_id, source_type)
);

-- Índices para consultas frecuentes
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_subreddit ON comments(subreddit);
CREATE INDEX IF NOT EXISTS idx_comments_created ON comments(created_utc);
CREATE INDEX IF NOT EXISTS idx_preprocessed_source ON preprocessed_texts(source_id, source_type);
CREATE INDEX IF NOT EXISTS idx_preprocessed_subreddit ON preprocessed_texts(subreddit);
CREATE INDEX IF NOT EXISTS idx_preprocessed_valid ON preprocessed_texts(is_valid);

-- Resultados de análisis de sentimiento (Agente ReAct)
CREATE TABLE IF NOT EXISTS sentiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    preprocessed_text_id INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    -- Clasificación RoBERTa
    roberta_label TEXT NOT NULL,          -- positive / negative / neutral
    roberta_confidence REAL NOT NULL,
    -- Razonamiento ReAct
    decision TEXT NOT NULL,               -- accepted / cross_validated / ambiguous
    -- Clasificación final
    final_label TEXT NOT NULL,            -- positive / negative / neutral / ambiguous
    final_confidence REAL NOT NULL,
    -- VADER (solo si decision = 'cross_validated')
    vader_compound REAL,
    vader_label TEXT,
    -- Metadatos
    analyzed_at TEXT NOT NULL,
    FOREIGN KEY (preprocessed_text_id) REFERENCES preprocessed_texts(id),
    UNIQUE(source_id, source_type)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_subreddit ON sentiment_results(subreddit);
CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_results(final_label);
CREATE INDEX IF NOT EXISTS idx_sentiment_decision ON sentiment_results(decision);

-- Asignación de tópicos por texto (Agente de Tendencias)
CREATE TABLE IF NOT EXISTS topic_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    preprocessed_text_id INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    created_utc REAL NOT NULL,             -- Fecha del texto original (unix ts)
    topic_id INTEGER NOT NULL,             -- -1 = outlier en BERTopic
    topic_label TEXT,                      -- Palabras clave del tópico (ej: "trump_tariff_trade")
    topic_probability REAL,                -- Probabilidad de asignación
    model_run_id TEXT NOT NULL,            -- UUID del run de BERTopic (para multi-run)
    assigned_at TEXT NOT NULL,
    FOREIGN KEY (preprocessed_text_id) REFERENCES preprocessed_texts(id),
    UNIQUE(source_id, source_type, model_run_id)
);

CREATE INDEX IF NOT EXISTS idx_topic_assignments_topic ON topic_assignments(topic_id, model_run_id);
CREATE INDEX IF NOT EXISTS idx_topic_assignments_created ON topic_assignments(created_utc);
CREATE INDEX IF NOT EXISTS idx_topic_assignments_run ON topic_assignments(model_run_id);

-- Análisis de tendencias por tópico (resultados del agente ReAct)
CREATE TABLE IF NOT EXISTS trend_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id TEXT NOT NULL,
    topic_id INTEGER NOT NULL,
    topic_label TEXT,
    -- Métricas temporales
    current_weight REAL NOT NULL,          -- Peso en ventana actual (textos/total)
    historical_mean REAL NOT NULL,         -- Media histórica de peso
    historical_std REAL NOT NULL,          -- Desv. estándar histórica (raw, antes de floor)
    effective_std REAL NOT NULL,           -- Desv. estándar usada (con floor aplicado)
    delta REAL NOT NULL,                   -- Δ = (current - mean) / effective_std
    -- Ventanas temporales
    current_window_start TEXT NOT NULL,    -- ISO date
    current_window_end TEXT NOT NULL,
    historical_window_start TEXT NOT NULL,
    historical_window_end TEXT NOT NULL,
    n_current_texts INTEGER NOT NULL,      -- Textos en ventana actual con este tópico
    n_historical_texts INTEGER NOT NULL,   -- Textos históricos con este tópico
    corpus_coverage REAL NOT NULL,         -- % del corpus total cubierto por tópico
    -- Datos para evaluación de tendencia moderada (3+ días crecientes)
    consecutive_growth_days INTEGER DEFAULT 0,
    -- Decisión ReAct
    trend_decision TEXT NOT NULL,          -- emerging_trend / localized_spike / moderate_trend / discarded
    trend_reason TEXT NOT NULL,            -- Razón textual de la decisión
    -- Evolución temporal diaria (JSON: {"2026-02-20": 0.12, "2026-02-21": 0.15, ...})
    daily_weights_json TEXT,
    -- Metadatos
    analyzed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trend_run ON trend_analysis(model_run_id);
CREATE INDEX IF NOT EXISTS idx_trend_decision ON trend_analysis(trend_decision);
CREATE INDEX IF NOT EXISTS idx_trend_delta ON trend_analysis(delta DESC);

-- Ground truth: etiquetas de sentimiento generadas por LLM
CREATE TABLE IF NOT EXISTS ground_truth_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    original_text TEXT NOT NULL,
    llm_label TEXT NOT NULL,        -- positive / negative / neutral
    llm_reasoning TEXT,             -- justificación breve del LLM
    model_used TEXT NOT NULL,       -- modelo LLM usado (ej: deepseek-chat)
    labeled_at TEXT NOT NULL,
    UNIQUE(source_id, source_type)
);

CREATE INDEX IF NOT EXISTS idx_gt_label ON ground_truth_labels(llm_label);

-- Metadatos de recolecciones
CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subreddit TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    posts_collected INTEGER DEFAULT 0,
    comments_collected INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',     -- running, completed, failed
    parameters TEXT                     -- JSON con parámetros usados
);
"""
