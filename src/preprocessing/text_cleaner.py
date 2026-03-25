"""
Limpieza y normalización de texto para contenido de Reddit.
Proporciona dos niveles de preprocesamiento:
  - Sentimiento (RoBERTa): texto limpio y conciso, mantiene estructura emocional.
  - Tópicos (BERTopic): texto más completo con contexto, enfocado en semántica.
"""

import re
import unicodedata


class TextCleaner:
    """Limpieza de texto de Reddit con dos modos de salida."""

    # Patrones de regex compilados para eficiencia
    URL_PATTERN = re.compile(
        r"https?://\S+|www\.\S+|[\w.-]+\.(?:com|org|net|edu|gov|io|co)\b\S*",
        re.IGNORECASE,
    )
    REDDIT_LINK_PATTERN = re.compile(r"/r/\w+|/u/\w+")
    MENTION_PATTERN = re.compile(r"u/\w+|(?<!\w)@\w+")
    MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
    MARKDOWN_FORMATTING_PATTERN = re.compile(r"[*_~`]{1,3}")
    MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,6}\s*", re.MULTILINE)
    BLOCKQUOTE_PATTERN = re.compile(r"^>\s*", re.MULTILINE)
    HTML_ENTITY_PATTERN = re.compile(r"&\w+;|&#\d+;")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    REPEATED_PUNCT_PATTERN = re.compile(r"([!?.])\1{2,}")
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    # Contenido de bots y auto-moderadores
    BOT_PATTERNS = [
        re.compile(r"I am a bot", re.IGNORECASE),
        re.compile(r"this action was performed automatically", re.IGNORECASE),
        re.compile(r"please contact the moderators", re.IGNORECASE),
        re.compile(r"^As a reminder.*this subreddit", re.IGNORECASE | re.DOTALL),
    ]

    def is_bot_content(self, text: str) -> bool:
        """Detecta si el texto es generado por un bot o auto-moderador."""
        return any(p.search(text) for p in self.BOT_PATTERNS)

    def clean_base(self, text: str, url_replace: str = "", mention_replace: str = "") -> str:
        """
        Limpieza base aplicada a ambos modos.
        Reemplaza/elimina URLs, menciones, formato markdown y normaliza espacios.

        Args:
            url_replace: Texto de reemplazo para URLs (default: eliminar).
            mention_replace: Texto de reemplazo para menciones (default: eliminar).
        """
        if not text or not text.strip():
            return ""

        # Reemplazar markdown links por su texto visible
        text = self.MARKDOWN_LINK_PATTERN.sub(r"\1", text)

        # URLs → reemplazar o eliminar
        text = self.URL_PATTERN.sub(url_replace, text)

        # Links de Reddit (/r/sub, /u/user)
        text = self.REDDIT_LINK_PATTERN.sub("", text)

        # Menciones → reemplazar o eliminar
        text = self.MENTION_PATTERN.sub(mention_replace, text)

        # Eliminar formato markdown (negritas, cursivas, etc.)
        text = self.MARKDOWN_FORMATTING_PATTERN.sub("", text)
        text = self.MARKDOWN_HEADER_PATTERN.sub("", text)
        text = self.BLOCKQUOTE_PATTERN.sub("", text)

        # Eliminar entidades HTML
        text = self.HTML_ENTITY_PATTERN.sub(" ", text)

        # Normalizar unicode (caracteres acentuados y especiales)
        text = unicodedata.normalize("NFKC", text)

        # Normalizar puntuación repetida (!!!! -> !, ??? -> ?)
        text = self.REPEATED_PUNCT_PATTERN.sub(r"\1", text)

        # Normalizar espacios en blanco
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def clean_for_sentiment(self, text: str) -> str:
        """
        Preprocesamiento optimizado para análisis de sentimiento con RoBERTa.

        Basado en el paper de RoBERTa y convenciones de cardiffnlp:
        - NO lowercase (byte-level BPE fue entrenado con texto natural)
        - URLs → "http" (placeholder esperado por cardiffnlp)
        - Menciones → "@user" (placeholder esperado por cardiffnlp)
        - Mantiene signos de puntuación emocionales (!, ?)
        - Mantiene negaciones (crucial para sentimiento)
        - Mantiene números (pueden tener carga emocional en contexto político)
        - Elimina emojis (RoBERTa no los procesa bien)
        """
        text = self.clean_base(text, url_replace="http", mention_replace="@user")
        if not text:
            return ""

        # Eliminar emojis
        text = self.EMOJI_PATTERN.sub("", text)

        # Limpiar espacios resultantes
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def clean_for_topics(self, text: str) -> str:
        """
        Preprocesamiento optimizado para detección de tópicos con BERTopic.

        Características:
        - Mantiene el caso original (los embeddings capturan semántica mejor así)
        - Mantiene más contexto (números, nombres propios son relevantes)
        - Elimina emojis
        - Menos agresivo: preserva información semántica rica
        """
        text = self.clean_base(text)
        if not text:
            return ""

        # Eliminar emojis
        text = self.EMOJI_PATTERN.sub("", text)

        # Limpiar espacios resultantes
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def word_count(self, text: str) -> int:
        """Cuenta palabras en el texto."""
        if not text or not text.strip():
            return 0
        return len(text.split())
