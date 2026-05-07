"""
Agente ReAct de Análisis de Sentimiento.

Ciclo ReAct:
  Observación  → Lee textos preprocesados sin analizar desde la BD
  Razonamiento → Evalúa confianza de RoBERTa y decide la acción
  Acción       → Acepta / valida cruzado con Gemini LLM / rescata / marca ambiguo
  Registro     → Persiste resultados y decisiones en sentiment_results
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Optional

from loguru import logger

from src.database.db_manager import DatabaseManager

# Umbrales configurables
HIGH_CONF_THRESHOLD = 0.85
LOW_CONF_THRESHOLD = 0.50
MID_CONF_THRESHOLD = 0.65  # Para desempate en zona media

# Modelo RoBERTa preentrenado en texto de redes sociales
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Gemini LLM para cross-validación
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BATCH_SIZE = 20  # Textos por llamada a Gemini

GEMINI_PROMPT = """You are a political sentiment classifier for Reddit comments from r/politics.

Classify each comment as: positive, negative, neutral, or ambiguous.

Classification rules:
1. NEGATIVE: Clear criticism, anger, frustration, sarcasm attacking someone/something, cynicism about politics or politicians.
2. POSITIVE: Explicit endorsement, celebration, or genuine enthusiasm about a political outcome or figure. Do NOT classify as positive just because a comment contains words like "hoping", "better", "support", or "good" — these are often neutral in political context.
3. NEUTRAL: Questions, factual observations, analogies, metaphors describing situations, calls to action, pragmatic assessments, mixed opinions, administrative messages.
4. AMBIGUOUS: Only when genuinely impossible to classify. Very rare.

Critical guidance:
- "You mean [damaging fact]..." or "You mean since [event]..." = rhetorical sarcasm implying criticism → NEGATIVE.
- Analogies comparing political situations to everyday scenarios ("same energy as X") = NEUTRAL commentary, not negative.
- "We need X" / "We should X" (calls to action) = NEUTRAL unless accompanied by insults or profanity.
- "Hoping for X" / "that's probably better" = NEUTRAL pragmatic observation, NOT positive.
- When in doubt between negative and neutral → NEUTRAL.
- Explicit mockery of a specific person or policy → NEGATIVE.
- Moderation/admin messages → NEUTRAL.

Respond with ONLY a valid JSON array. Each element:
{"id": <number>, "label": "<positive|negative|neutral|ambiguous>", "reasoning": "<one sentence>"}

Texts:
"""

def _are_opposite_labels(label1: str, label2: str) -> bool:
    """Verifica si dos labels son opuestos (positive vs negative)."""
    return (
        (label1 == "positive" and label2 == "negative")
        or (label1 == "negative" and label2 == "positive")
    )


class SentimentAgent:
    """
    Agente de análisis de sentimiento con patrón ReAct.

    Decisiones:
      - confianza > HIGH_CONF          → 'accepted'       (RoBERTa directo)
      - LOW_CONF < confianza ≤ HIGH_CONF → cross-validación con Gemini LLM
      - confianza ≤ LOW_CONF           → rescate con Gemini LLM o 'ambiguous'
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        high_conf: float = HIGH_CONF_THRESHOLD,
        low_conf: float = LOW_CONF_THRESHOLD,
        mid_conf: float = MID_CONF_THRESHOLD,
    ):
        self.db = db or DatabaseManager()
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.mid_conf = mid_conf
        self._roberta = None
        self._gemini_client = None

    # --- Carga de modelos (lazy) ---

    def _load_roberta(self):
        if self._roberta is None:
            from transformers import pipeline
            logger.info(f"Cargando RoBERTa ({ROBERTA_MODEL})...")
            self._roberta = pipeline(
                "text-classification",
                model=ROBERTA_MODEL,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("RoBERTa cargado.")

    def _get_gemini(self):
        if self._gemini_client is None:
            from google import genai
            api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            self._gemini_client = genai.Client(api_key=api_key)
            logger.info(f"Gemini client inicializado ({GEMINI_MODEL}).")
        return self._gemini_client

    # --- Gemini LLM classification ---

    def _classify_with_gemini(self, texts: list[str]) -> list[dict]:
        """Clasifica un lote de textos con Gemini LLM.

        Returns:
            Lista de {"label": str, "reasoning": str} por cada texto.
        """
        client = self._get_gemini()

        prompt = GEMINI_PROMPT
        for i, text in enumerate(texts, 1):
            prompt += f'\n{i}. "{text[:500]}"\n'

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                raw = response.text.strip()
                raw = re.sub(r'^```json\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

                try:
                    results = json.loads(raw)
                except json.JSONDecodeError:
                    match = re.search(r'\[.*\]', raw, re.DOTALL)
                    if match:
                        results = json.loads(match.group())
                    else:
                        raise ValueError(f"No se pudo parsear JSON de Gemini: {raw[:200]}")

                # Validar que tengamos el número correcto de resultados
                if len(results) != len(texts):
                    logger.warning(
                        f"[Gemini] Esperaba {len(texts)} resultados, recibió {len(results)}. "
                        f"Usando lo disponible."
                    )

                return [
                    {
                        "label": r.get("label", "ambiguous").lower(),
                        "reasoning": r.get("reasoning", ""),
                    }
                    for r in results
                ]

            except Exception as e:
                logger.warning(f"[Gemini] Intento {attempt + 1}/{max_retries} falló: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    logger.error(f"[Gemini] Todos los intentos fallaron para lote de {len(texts)} textos.")
                    return [{"label": "ambiguous", "reasoning": "Gemini API error"} for _ in texts]

    # --- ReAct: Observación ---

    def _observe(self, limit: int) -> list[dict]:
        texts = self.db.get_unanalyzed_texts_for_sentiment(limit=limit)
        logger.info(f"[Observación] {len(texts)} textos pendientes de análisis.")
        return texts

    # --- ReAct: Razonamiento ---

    def _reason(self, roberta_scores: list[dict]) -> tuple[str, str, float]:
        """
        Dado el output de RoBERTa, determina la decisión inicial y extrae la predicción.
        Returns: (decision, roberta_label, roberta_confidence)
        """
        best = max(roberta_scores, key=lambda x: x["score"])
        label = best["label"].lower()
        confidence = best["score"]

        if confidence > self.high_conf:
            decision = "accepted"
        elif confidence > self.low_conf:
            decision = "needs_cross_validation"
        else:
            decision = "needs_rescue"

        logger.debug(
            f"[Razonamiento] {label} ({confidence:.3f}) → {decision}"
        )
        return decision, label, confidence

    # --- ReAct: Acción ---

    def _act_with_gemini(
        self, roberta_label: str, roberta_confidence: float, gemini_label: str,
    ) -> dict:
        """
        Decide el label final combinando RoBERTa y Gemini.

        Lógica:
          conf > 0.85 → accepted (no llega aquí)
          conf 0.50-0.85:
            acuerdo → cross_validated
            desacuerdo + conf > 0.65 → roberta gana
            desacuerdo + conf ≤ 0.65 → gemini gana (si no son opuestos)
          conf ≤ 0.50:
            acuerdo → rescued
            gemini dice ambiguous → ambiguous
            labels opuestos → ambiguous
            gemini da label claro → rescued
        """
        gemini_label = gemini_label.lower()

        if roberta_confidence > self.low_conf:
            # Zona media (0.50 - 0.85)
            if roberta_label == gemini_label:
                return {
                    "decision": "cross_validated",
                    "final_label": roberta_label,
                    "final_confidence": min(roberta_confidence + 0.05, 1.0),
                    "gemini_label": gemini_label,
                }
            elif roberta_confidence > self.mid_conf:
                return {
                    "decision": "cross_validated",
                    "final_label": roberta_label,
                    "final_confidence": roberta_confidence,
                    "gemini_label": gemini_label,
                }
            else:
                # conf ≤ 0.65, Gemini gana (salvo opuestos)
                if _are_opposite_labels(roberta_label, gemini_label):
                    return {
                        "decision": "ambiguous",
                        "final_label": "ambiguous",
                        "final_confidence": roberta_confidence,
                        "gemini_label": gemini_label,
                    }
                if gemini_label == "ambiguous":
                    return {
                        "decision": "ambiguous",
                        "final_label": "ambiguous",
                        "final_confidence": roberta_confidence,
                        "gemini_label": gemini_label,
                    }
                return {
                    "decision": "cross_validated",
                    "final_label": gemini_label,
                    "final_confidence": roberta_confidence,
                    "gemini_label": gemini_label,
                }
        else:
            # Zona baja (≤ 0.50) — rescate
            if roberta_label == gemini_label:
                return {
                    "decision": "rescued",
                    "final_label": roberta_label,
                    "final_confidence": min(roberta_confidence + 0.05, 1.0),
                    "gemini_label": gemini_label,
                }
            if gemini_label == "ambiguous":
                return {
                    "decision": "ambiguous",
                    "final_label": "ambiguous",
                    "final_confidence": roberta_confidence,
                    "gemini_label": gemini_label,
                }
            if _are_opposite_labels(roberta_label, gemini_label):
                return {
                    "decision": "ambiguous",
                    "final_label": "ambiguous",
                    "final_confidence": roberta_confidence,
                    "gemini_label": gemini_label,
                }
            # Gemini da label claro y no son opuestos
            return {
                "decision": "rescued",
                "final_label": gemini_label,
                "final_confidence": roberta_confidence,
                "gemini_label": gemini_label,
            }

    # --- ReAct: Registro ---

    def _record(self, results: list[dict]) -> int:
        inserted = self.db.insert_sentiment_batch(results)
        logger.info(f"[Registro] {inserted} resultados guardados en BD.")
        return inserted

    # --- Ciclo principal ---

    def run(self, limit: int = 1000, batch_size: int = 64) -> dict:
        """
        Ejecuta el ciclo ReAct completo.

        Args:
            limit:      Máximo de textos a procesar en esta ejecución.
            batch_size: Tamaño de lote para inferencia con RoBERTa.

        Returns:
            dict con métricas del ciclo.
        """
        self._load_roberta()

        # Observación
        texts = self._observe(limit)
        if not texts:
            logger.info("No hay textos pendientes de análisis de sentimiento.")
            return {"total": 0}

        logger.info(f"Iniciando análisis de {len(texts)} textos (batch_size={batch_size})...")

        results = []
        counters = {"accepted": 0, "cross_validated": 0, "rescued": 0, "ambiguous": 0}

        # Paso 1: RoBERTa en lotes
        roberta_results = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start: batch_start + batch_size]
            input_texts = [t["text_for_sentiment"] for t in batch]
            roberta_outputs = self._roberta(input_texts)

            for text_row, roberta_scores in zip(batch, roberta_outputs):
                decision, roberta_label, roberta_confidence = self._reason(roberta_scores)
                roberta_results.append({
                    "text_row": text_row,
                    "decision": decision,
                    "roberta_label": roberta_label,
                    "roberta_confidence": roberta_confidence,
                })

            logger.info(
                f"  RoBERTa lote {batch_start // batch_size + 1}: "
                f"{min(batch_start + batch_size, len(texts))}/{len(texts)}"
            )

        # Paso 2: Gemini para textos que lo necesitan
        needs_gemini = [
            (i, r) for i, r in enumerate(roberta_results)
            if r["decision"] in ("needs_cross_validation", "needs_rescue")
        ]

        gemini_results_map = {}
        if needs_gemini:
            logger.info(f"[Gemini] Clasificando {len(needs_gemini)} textos con LLM...")
            for gem_start in range(0, len(needs_gemini), GEMINI_BATCH_SIZE):
                gem_batch = needs_gemini[gem_start: gem_start + GEMINI_BATCH_SIZE]
                gem_texts = [r["text_row"]["text_for_sentiment"] for _, r in gem_batch]

                gem_labels = self._classify_with_gemini(gem_texts)

                for (orig_idx, _), gem_result in zip(gem_batch, gem_labels):
                    gemini_results_map[orig_idx] = gem_result

                logger.info(
                    f"  Gemini lote {gem_start // GEMINI_BATCH_SIZE + 1}: "
                    f"{min(gem_start + GEMINI_BATCH_SIZE, len(needs_gemini))}/{len(needs_gemini)}"
                )
                # Rate limiting
                time.sleep(0.5)

        # Paso 3: Combinar resultados
        for i, r in enumerate(roberta_results):
            text_row = r["text_row"]
            roberta_label = r["roberta_label"]
            roberta_confidence = r["roberta_confidence"]

            if r["decision"] == "accepted":
                final_label = roberta_label
                final_confidence = roberta_confidence
                decision = "accepted"
                gemini_label = None
            else:
                gem = gemini_results_map.get(i, {"label": "ambiguous", "reasoning": ""})
                action = self._act_with_gemini(roberta_label, roberta_confidence, gem["label"])
                final_label = action["final_label"]
                final_confidence = action["final_confidence"]
                decision = action["decision"]
                gemini_label = action.get("gemini_label")

            counters[decision] = counters.get(decision, 0) + 1

            results.append({
                "preprocessed_text_id": text_row["id"],
                "source_id": text_row["source_id"],
                "source_type": text_row["source_type"],
                "subreddit": text_row["subreddit"],
                "roberta_label": roberta_label,
                "roberta_confidence": roberta_confidence,
                "decision": decision,
                "final_label": final_label,
                "final_confidence": final_confidence,
                "gemini_label": gemini_label,
                "analyzed_at": datetime.utcnow().isoformat(),
            })

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
