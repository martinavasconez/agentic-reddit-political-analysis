"""
Cliente de Reddit API usando PRAW.
Maneja la conexión y provee acceso al API de forma centralizada.
"""

import praw
from loguru import logger

from config.settings import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT


def create_reddit_client() -> praw.Reddit:
    """Crea y retorna una instancia autenticada de Reddit (read-only)."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise ValueError(
            "Faltan credenciales de Reddit. "
            "Configura REDDIT_CLIENT_ID y REDDIT_CLIENT_SECRET en el archivo .env"
        )

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    # Verificar que la conexión es read-only (no necesitamos escribir)
    if not reddit.read_only:
        logger.warning("La conexión no es read-only. Se esperaba acceso de solo lectura.")

    logger.info(f"Cliente de Reddit conectado (read_only={reddit.read_only})")
    return reddit


