"""
Agente ReAct de Análisis de Sentimiento.

Ciclo ReAct:
  Observación  → Lee textos preprocesados sin analizar desde la BD
  Razonamiento → Evalúa confianza de RoBERTa y decide la acción
  Acción       → Acepta / valida cruzado con VADER / marca ambiguo
  Registro     → Persiste resultados y decisiones en sentiment_results
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.database.db_manager import DatabaseManager

# Umbrales configurables
HIGH_CONF_THRESHOLD = 0.85
LOW_CONF_THRESHOLD = 0.50

# Modelo RoBERTa preentrenado en texto de redes sociales
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _vader_to_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


class SentimentAgent:
    """
    Agente de análisis de sentimiento con patrón ReAct.

    Decisiones:
      - confianza > HIGH_CONF          → 'accepted'       (RoBERTa directo)
      - LOW_CONF < confianza ≤ HIGH_CONF → 'cross_validated' (valida con VADER)
      - confianza ≤ LOW_CONF           → 'ambiguous'      (excluye del agregado)
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        high_conf: float = HIGH_CONF_THRESHOLD,
        low_conf: float = LOW_CONF_THRESHOLD,
    ):
        self.db = db or DatabaseManager()
        self.high_conf = high_conf
        self.low_conf = low_conf
        self._roberta = None
        self._vader = None

    #  Carga de modelos (lazy)
    def _load_models(self):
        if self._roberta is None:
            from transformers import pipeline
            logger.info(f"Cargando RoBERTa ({ROBERTA_MODEL})...")
            self._roberta = pipeline(
                "text-classification",
                model=ROBERTA_MODEL,
                top_k=None,          # devuelve scores para todas las clases
                truncation=True,
                max_length=512,
            )
            logger.info("RoBERTa cargado.")

        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER cargado.")

    #  ReAct: Observación
    def _observe(self, limit: int) -> list[dict]:
        texts = self.db.get_unanalyzed_texts_for_sentiment(limit=limit)
        logger.info(f"[Observación] {len(texts)} textos pendientes de análisis.")
        return texts

    #  ReAct: Razonamiento
    def _reason(self, roberta_scores: list[dict]) -> tuple[str, str, float]:
        """
        Dado el output de RoBERTa (lista de {label, score}), determina la decisión y extrae la predicción principal.
        Returns: (decision, roberta_label, roberta_confidence)
        """
        best = max(roberta_scores, key=lambda x: x["score"])
        label = best["label"].lower()      # positive / negative / neutral
        confidence = best["score"]

        if confidence > self.high_conf:
            decision = "accepted"
        elif confidence > self.low_conf:
            decision = "cross_validated"
        else:
            decision = "ambiguous"

        logger.debug(
            f"[Razonamiento] {label} ({confidence:.3f}) → {decision}"
        )
        return decision, label, confidence

    #  ReAct: Acción
    def _act(self, text: str, decision: str, roberta_label: str,
roberta_confidence: float,
    ) -> dict:
        """
        Ejecuta la acción correspondiente a la decisión tomada.

        Returns:
            dict con final_label, final_confidence y datos VADER si aplica.
        """
        vader_compound = None
        vader_label = None

        if decision == "accepted":
            final_label = roberta_label
            final_confidence = roberta_confidence

        elif decision == "cross_validated":
            scores = self._vader.polarity_scores(text)
            vader_compound = scores["compound"]
            vader_label = _vader_to_label(vader_compound)

            if vader_label == roberta_label:
                # Ambos coinciden → boost leve de confianza
                final_label = roberta_label
                final_confidence = min(roberta_confidence + 0.05, 1.0)
            else:
                # Discrepan → RoBERTa gana (mejor contexto político)
                # VADER queda registrado como evidencia de ambigüedad
                final_label = roberta_label
                final_confidence = roberta_confidence
            logger.debug(
                f"[Acción] Cross-validate: RoBERTa={roberta_label} | "
                f"VADER={vader_label} (compound={vader_compound:.3f}) → {final_label}"
            )

        else:  # ambiguous
            final_label = "ambiguous"
            final_confidence = roberta_confidence

        return {
            "final_label": final_label,
            "final_confidence": final_confidence,
            "vader_compound": vader_compound,
            "vader_label": vader_label,
        }

    #  ReAct: Registro
    def _record(self, results: list[dict]) -> int:
        inserted = self.db.insert_sentiment_batch(results)
        logger.info(f"[Registro] {inserted} resultados guardados en BD.")
        return inserted

    #  Ciclo principal
    def run(self, limit: int = 1000, batch_size: int = 64) -> dict:
        """
        Ejecuta el ciclo ReAct completo.

        Args:
            limit:      Máximo de textos a procesar en esta ejecución.
            batch_size: Tamaño de lote para inferencia con RoBERTa.

        Returns:
            dict con métricas del ciclo.
        """
        self._load_models()

        # Observación
        texts = self._observe(limit)
        if not texts:
            logger.info("No hay textos pendientes de análisis de sentimiento.")
            return {"total": 0}

        logger.info(f"Iniciando análisis de {len(texts)} textos (batch_size={batch_size})...")

        results = []
        counters = {"accepted": 0, "cross_validated": 0, "ambiguous": 0}

        # Inferencia en lotes para eficiencia
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start: batch_start + batch_size]
            input_texts = [t["text_for_sentiment"] for t in batch]

            # RoBERTa en lote
            roberta_outputs = self._roberta(input_texts)

            for text_row, roberta_scores in zip(batch, roberta_outputs):
                # Razonamiento
                decision, roberta_label, roberta_confidence = self._reason(roberta_scores)

                # Acción
                action_result = self._act(
                    text_row["text_for_sentiment"],
                    decision,
                    roberta_label,
                    roberta_confidence,
                )

                counters[decision] += 1

                results.append({
                    "preprocessed_text_id": text_row["id"],
                    "source_id": text_row["source_id"],
                    "source_type": text_row["source_type"],
                    "subreddit": text_row["subreddit"],
                    "roberta_label": roberta_label,
                    "roberta_confidence": roberta_confidence,
                    "decision": decision,
                    "final_label": action_result["final_label"],
                    "final_confidence": action_result["final_confidence"],
                    "vader_compound": action_result["vader_compound"],
                    "vader_label": action_result["vader_label"],
                    "analyzed_at": datetime.utcnow().isoformat(),
                })

            logger.info(
                f"  Lote {batch_start // batch_size + 1}: "
                f"{min(batch_start + batch_size, len(texts))}/{len(texts)} textos procesados"
            )

        # Registro
        self._record(results)

        # Métricas del ciclo
        total = len(results)
        label_dist = {}
        for r in results:
            label_dist[r["final_label"]] = label_dist.get(r["final_label"], 0) + 1

        avg_conf = sum(r["final_confidence"] for r in results) / total if total else 0
        pct_ambiguous = (label_dist.get("ambiguous", 0) / total * 100) if total else 0

        summary = {
            "total": total,
            "decisions": counters,
            "label_distribution": label_dist,
            "avg_confidence": round(avg_conf, 4),
            "pct_ambiguous": round(pct_ambiguous, 2),
        }

        logger.info("=" * 50)
        logger.info("RESUMEN AGENTE DE SENTIMIENTO")
        logger.info(f"  Total analizados : {total}")
        logger.info(f"  Decisiones       : {counters}")
        logger.info(f"  Distribución     : {label_dist}")
        logger.info(f"  Confianza prom.  : {summary['avg_confidence']}")
        logger.info(f"  % Ambiguos       : {summary['pct_ambiguous']}%")
        logger.info("=" * 50)

        return summary
