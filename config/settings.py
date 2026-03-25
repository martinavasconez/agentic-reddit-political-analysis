"""
Configuración centralizada del proyecto.
Carga credenciales desde .env y define parámetros globales.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Reddit API
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "tesis/0.1")

# DeepSeek API (ground truth labeling)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Subreddits objetivo
TARGET_SUBREDDITS = ["politics"]

# Base de datos
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "reddit_political.db"

# Recolección (PRAW)
DEFAULT_COLLECTION_DAYS = 7       # Ventana PRAW
POSTS_PER_SUBREDDIT = 500         # Máximo de posts por subreddit por extracción
COMMENTS_PER_POST = 100           # Máximo de comentarios por post vía PRAW
RATE_LIMIT_SLEEP = 1              # Segundos entre requests a la API de Reddit

# Preprocesamiento
MIN_WORD_COUNT = 10               # Mínimo de palabras para texto válido (sentimiento)
MAX_TEXT_LENGTH = 10000           # Máximo de caracteres por texto (trunca si supera)
ROBERTA_MAX_TOKENS = 512          # Límite de tokens para RoBERTa
BERTOPIC_MIN_WORDS = 15           # Mínimo de palabras para BERTopic (requiere más contexto)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")