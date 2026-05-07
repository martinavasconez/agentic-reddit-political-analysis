"""
Pipeline Tradicional (sin comportamiento agentic).

Ejecuta los mismos pasos del sistema agentic pero sin:
  - Umbrales de confianza
  - Cross-validation con Gemini LLM
  - Abstención (no hay categoría ambiguo)
  - Decisiones condicionales en tendencias (reporta todo sin filtrar)

Esto permite comparar directamente el impacto del diseño agentic.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger

from src.database.db_manager import DatabaseManager
from src.agents.validation.report_generator import ReportGenerator

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class TraditionalPipeline:
    """Pipeline secuencial sin decisiones agentic."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self._roberta = None
        self._topic_model = None
        self.run_id = str(uuid.uuid4())[:8]

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

    def _load_bertopic(self):
        if self._topic_model is None:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer

            logger.info("Cargando BERTopic...")
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                min_df=3,
                max_df=0.85,
            )
            self._topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer,
                nr_topics=None,
                min_topic_size=50,
                calculate_probabilities=False,
                verbose=False,
            )
            logger.info("BERTopic cargado.")

    # ------------------------------------------------------------------
    # Sentimiento: RoBERTa directo sin umbrales
    # ------------------------------------------------------------------

    def run_sentiment(self, limit: int = 1000, batch_size: int = 64) -> dict:
        """Clasifica sentimiento con RoBERTa argmax directo, sin Gemini ni abstención."""
        self._load_roberta()

        texts = self.db.get_unanalyzed_texts_for_pipeline(limit=limit)
        if not texts:
            logger.info("No hay textos pendientes para pipeline tradicional.")
            return {"total": 0}

        logger.info(f"[Pipeline] Clasificando {len(texts)} textos (RoBERTa directo)...")

        results = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start: batch_start + batch_size]
            input_texts = [t["text_for_sentiment"] for t in batch]
            roberta_outputs = self._roberta(input_texts)

            for text_row, scores in zip(batch, roberta_outputs):
                best = max(scores, key=lambda x: x["score"])
                results.append({
                    "preprocessed_text_id": text_row["id"],
                    "source_id": text_row["source_id"],
                    "source_type": text_row["source_type"],
                    "subreddit": text_row["subreddit"],
                    "roberta_label": best["label"].lower(),
                    "roberta_confidence": best["score"],
                    "analyzed_at": datetime.utcnow().isoformat(),
                })

            logger.info(
                f"  Lote {batch_start // batch_size + 1}: "
                f"{min(batch_start + batch_size, len(texts))}/{len(texts)}"
            )

        inserted = self.db.insert_pipeline_results_batch(results)
        logger.info(f"[Pipeline] {inserted} resultados de sentimiento guardados.")

        label_dist = {}
        for r in results:
            label_dist[r["roberta_label"]] = label_dist.get(r["roberta_label"], 0) + 1

        return {
            "total": len(results),
            "inserted": inserted,
            "label_distribution": label_dist,
        }

    # ------------------------------------------------------------------
    # Tendencias: BERTopic + Δ sin decisiones de filtrado
    # ------------------------------------------------------------------

    def run_trends(self, limit: int = 50000, current_days: float = 2) -> dict:
        """Ejecuta BERTopic y reporta TODOS los tópicos con Δ sin filtrar."""
        self._load_bertopic()

        texts = self.db.get_texts_for_topic_modeling(limit=limit)
        if not texts:
            logger.info("No hay textos para modelado de tópicos.")
            return {"total": 0}

        timestamps = [t["created_utc"] for t in texts]
        max_ts = max(timestamps)
        cutoff = max_ts - (current_days * 86400)

        historical = [t for t in texts if t["created_utc"] < cutoff]
        current = [t for t in texts if t["created_utc"] >= cutoff]

        logger.info(
            f"[Pipeline] BERTopic sobre {len(texts)} textos "
            f"(histórico: {len(historical)}, actual: {len(current)})"
        )

        all_docs = [t["text_for_topics"] for t in texts]
        all_topic_ids, _ = self._topic_model.fit_transform(all_docs)

        n_topics = len(set(all_topic_ids)) - (1 if -1 in all_topic_ids else 0)
        logger.info(f"[Pipeline] {n_topics} tópicos detectados.")

        hist_ids = list(all_topic_ids[:len(historical)])
        curr_ids = list(all_topic_ids[len(historical):])

        # Calcular Δ para cada tópico (sin filtrar)
        from collections import defaultdict, Counter

        # Pesos diarios históricos
        hist_days = defaultdict(list)
        for t, tid in zip(historical, hist_ids):
            day = datetime.fromtimestamp(t["created_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
            hist_days[day].append(tid)

        # Peso actual por tópico
        curr_count = Counter(curr_ids)
        total_curr = len(curr_ids)
        total_all = len(all_docs)

        topic_info = self._topic_model.get_topic_info()
        topic_labels = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid != -1:
                topic_labels[tid] = row.get("Name", f"topic_{tid}")

        trend_results = []
        unique_topics = set(all_topic_ids) - {-1}

        for tid in unique_topics:
            daily_weights = {}
            for day, topics_in_day in sorted(hist_days.items()):
                total_day = len(topics_in_day)
                count_day = sum(1 for t in topics_in_day if t == tid)
                daily_weights[day] = count_day / total_day if total_day else 0

            weights = list(daily_weights.values())
            hist_mean = np.mean(weights) if weights else 0
            hist_std = np.std(weights) if weights else 0
            eff_std = max(hist_std, 0.005)

            curr_weight = curr_count.get(tid, 0) / total_curr if total_curr else 0
            delta = (curr_weight - hist_mean) / eff_std if eff_std > 0 else 0
            coverage = sum(1 for t in all_topic_ids if t == tid) / total_all

            trend_results.append({
                "topic_id": tid,
                "topic_label": topic_labels.get(tid, f"topic_{tid}"),
                "delta": round(delta, 4),
                "current_weight": round(curr_weight, 6),
                "historical_mean": round(hist_mean, 6),
                "historical_std": round(hist_std, 6),
                "effective_std": round(eff_std, 6),
                "coverage": round(coverage, 4),
                "n_current": curr_count.get(tid, 0),
                "daily_weights": daily_weights,
            })

        trend_results.sort(key=lambda x: x["delta"], reverse=True)
        logger.info(
            f"[Pipeline] Top 5 Δ: "
            + ", ".join(f"{t['topic_label'][:20]}(Δ={t['delta']:.2f})" for t in trend_results[:5])
        )

        return {
            "total_texts": len(texts),
            "n_topics": n_topics,
            "trends": trend_results,
            "run_id": self.run_id,
        }

    # ------------------------------------------------------------------
    # Pipeline completo con reporte
    # ------------------------------------------------------------------

    def run_full(
        self,
        limit_sentiment: int = 1000,
        limit_trends: int = 50000,
        current_days: float = 2,
        batch_size: int = 64,
    ) -> dict:
        """Ejecuta pipeline completo: sentimiento + tendencias + reporte."""
        logger.info("=" * 60)
        logger.info("PIPELINE TRADICIONAL — Ejecución completa")
        logger.info("=" * 60)

        sentiment_result = self.run_sentiment(limit=limit_sentiment, batch_size=batch_size)
        trends_result = self.run_trends(limit=limit_trends, current_days=current_days)

        # Generar reporte
        rg = ReportGenerator()

        if sentiment_result.get("label_distribution"):
            rg.plot_sentiment_distribution(
                sentiment_result["label_distribution"],
                title="Pipeline Tradicional: Sentimiento (RoBERTa directo)",
            )

        if trends_result.get("trends"):
            top_trends = [
                {"topic_label": t["topic_label"], "delta": t["delta"],
                 "trend_decision": "unfiltered"}
                for t in trends_result["trends"][:20]
            ]
            rg.plot_trend_deltas(top_trends, title="Pipeline Tradicional: Top 20 Δ")

        sections = [
            {
                "heading": "Sentimiento (RoBERTa directo, sin umbrales)",
                "text": (
                    f"Textos clasificados: {sentiment_result.get('total', 0):,}\n\n"
                    f"Distribución: {sentiment_result.get('label_distribution', {})}\n\n"
                    "**Sin Gemini, sin abstención, sin umbrales de confianza.**"
                ),
                "image": "sentiment_distribution.png" if sentiment_result.get("label_distribution") else None,
            },
            {
                "heading": "Tendencias (BERTopic + Δ, sin filtrado)",
                "text": (
                    f"Tópicos detectados: {trends_result.get('n_topics', 0)}\n\n"
                    f"Todos los tópicos reportados sin decisión agentic (sin emerging/spike/discarded)."
                ),
                "image": "trend_deltas.png" if trends_result.get("trends") else None,
            },
        ]
        report_path = rg.generate_markdown_report(sections, title="Reporte Pipeline Tradicional")

        logger.info("=" * 60)
        logger.info(f"Reporte: {report_path}")
        logger.info("=" * 60)

        return {
            "sentiment": sentiment_result,
            "trends": trends_result,
            "report_path": str(report_path),
        }
