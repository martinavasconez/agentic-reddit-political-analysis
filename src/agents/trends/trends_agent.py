"""
Agente ReAct de Detección de Tendencias.

Ciclo ReAct:
  Observación  → Carga textos preprocesados con timestamps desde la BD
  Razonamiento → BERTopic asigna tópicos; calcula Δ por ventana temporal
  Acción       → Clasifica cada tópico: emerging_trend / localized_spike /
                 moderate_trend / discarded
  Registro     → Persiste asignaciones de tópicos y resultados en BD

Métrica Δ:
    Δ = (w_current - mean_historical) / effective_std
    donde effective_std = max(historical_std, STD_FLOOR)
    para evitar inestabilidad con series históricas cortas.

Split temporal (según protocolo de evaluación):
    BERTopic se entrena con fit_transform() sobre TODOS los textos
    (históricos + evaluación combinados). Las asignaciones se separan
    post-hoc por timestamp para calcular Δ. Esto mejora la estabilidad
    de los tópicos respecto a entrenar solo con datos históricos, y es
    válido porque BERTopic es no supervisado (no tiene acceso a etiquetas
    temporales durante el entrenamiento).

    Ventana histórica: max_ts - CURRENT_DAYS hacia atrás (~82 días con corpus de 89 días)
    Ventana actual:    últimos CURRENT_DAYS días (~7 días)

Decisiones (umbrales calibrados con corpus real):
    Δ ≥ 1.5 y coverage > 5%  → emerging_trend
    Δ ≥ 1.5 y coverage ≤ 5%  → localized_spike
    1.0 ≤ Δ < 1.5 y 3+ días creciendo → moderate_trend
    1.0 ≤ Δ < 1.5 y decreciendo       → discarded (pico pasajero)
    Δ < 1.0                            → discarded

Nota sobre estabilidad estadística:
    Con 82 días de baseline la desviación estándar histórica se calcula
    sobre ~82 puntos diarios, lo cual es estadísticamente robusto.
    STD_FLOOR = 0.005 protege contra tópicos con varianza históricamente
    baja que generarían Δ artificialmente alto.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger

from src.database.db_manager import DatabaseManager

#  Umbrales y configuración
DELTA_HIGH = 1.5           # Umbral para tendencia/spike (calibrado con corpus real de 89 días)
DELTA_MODERATE = 1.0       # Umbral para tendencia moderada
COVERAGE_THRESHOLD = 0.05  # 5% del corpus = tendencia emergente vs spike localizado
STD_FLOOR = 0.005          # Floor mínimo para desv. estándar (evita Δ artificialmente alto)
MIN_TOPIC_TEXTS = 10       # Textos mínimos en ventana actual para evaluar un tópico

# Split temporal dinámico: cutoff = max_ts - CURRENT_DAYS
HISTORICAL_DAYS = 60       # Referencia de ventana histórica (con corpus de 89 días: ~82 días efectivos)
CURRENT_DAYS = 7           # Días de ventana de evaluación

# Stopwords custom: palabras genéricas de Reddit/discusión que no aportan semántica política
REDDIT_STOPWORDS = [
    "comment", "comments", "post", "posts", "response", "responses",
    "said", "says", "saying", "explained", "replied", "reply",
    "upvote", "downvote", "edit", "deleted", "removed",
    "thread", "subreddit", "reddit", "mod", "moderator",
    "yeah", "yes", "no", "ok", "okay", "lol", "lmao", "wtf",
    "thing", "things", "people", "person", "someone", "something",
    "way", "ways", "lot", "lots", "bit", "just", "like", "really",
    "actually", "basically", "literally", "pretty", "quite",
    "think", "thought", "know", "knew", "want", "wanted",
    "going", "gone", "come", "came", "make", "made", "take", "took",
]


class TrendsAgent:
    """
    Agente de detección de tendencias temáticas con patrón ReAct.

    Usa BERTopic para modelado temático y calcula Δ normalizado
    para comparar el peso actual de cada tópico contra su baseline histórico.
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        n_topics: Optional[int] = None,     # None = auto-detección BERTopic
        historical_days: int = HISTORICAL_DAYS,
        current_days: int = CURRENT_DAYS,
        delta_high: float = DELTA_HIGH,
        delta_moderate: float = DELTA_MODERATE,
        coverage_threshold: float = COVERAGE_THRESHOLD,
    ):
        self.db = db or DatabaseManager()
        self.n_topics = n_topics
        self.historical_days = historical_days
        self.current_days = current_days
        self.delta_high = delta_high
        self.delta_moderate = delta_moderate
        self.coverage_threshold = coverage_threshold
        self._topic_model = None
        self.model_run_id = str(uuid.uuid4())[:8]  # ID corto para este run

    #  Carga de modelos (lazy)

    def _load_models(self):
        if self._topic_model is None:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer

            logger.info("Cargando BERTopic con sentence-transformers...")
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            # Vectorizer con n-gramas 1-2, filtra stopwords inglés + palabras
            # genéricas de Reddit que no aportan semántica política
            all_stopwords = list(CountVectorizer(stop_words="english")
                                 .get_stop_words()) + REDDIT_STOPWORDS
            vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                stop_words=all_stopwords,
                min_df=3,
                max_df=0.85,
            )
            # n_topics=None → BERTopic auto-determina (HDBSCAN)
            # min_topic_size=50 con corpus de 200K da tópicos coherentes (~150-300)
            # calculate_probabilities=False: evita error de HDBSCAN KD-tree
            self._topic_model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer,
                nr_topics=self.n_topics,   # None = auto
                min_topic_size=50,
                calculate_probabilities=False,
                verbose=False,
            )
            logger.info("BERTopic cargado.")

    #  ReAct: Observación
    def _observe(self, limit: int) -> tuple[list[dict], list[dict]]:
        """
        Carga textos y aplica el split temporal:
          - Histórico (días 1-60): para BERTopic.fit() y baseline Δ
          - Evaluación (días 61-90): para BERTopic.transform() y cálculo de tendencia

        Returns:
            (historical_texts, current_texts)
        """
        texts = self.db.get_texts_for_topic_modeling(limit=limit)
        logger.info(f"[Observación] {len(texts)} textos cargados para modelado.")

        if not texts:
            return [], []

        timestamps = [t["created_utc"] for t in texts if t["created_utc"]]
        self._max_ts = max(timestamps)
        self._min_ts = min(timestamps)

        # Split: ventana actual = últimos current_days desde el máximo
        self._current_cutoff = self._max_ts - (self.current_days * 86400)

        historical = [t for t in texts if t["created_utc"] < self._current_cutoff]
        current = [t for t in texts if t["created_utc"] >= self._current_cutoff]

        max_date = datetime.fromtimestamp(self._max_ts, tz=timezone.utc).date()
        min_date = datetime.fromtimestamp(self._min_ts, tz=timezone.utc).date()
        cutoff_date = datetime.fromtimestamp(self._current_cutoff, tz=timezone.utc).date()

        logger.info(
            f"[Observación] Split temporal — "
            f"Histórico: {min_date} → {cutoff_date} ({len(historical)} textos, ~días 1-{self.historical_days}) | "
            f"Evaluación: {cutoff_date} → {max_date} ({len(current)} textos, ~días 61-90)"
        )

        if len(historical) < 100:
            logger.warning(
                f"[Observación] Solo {len(historical)} textos históricos. "
                f"Se recomienda ≥ 1000 para BERTopic estable."
            )

        return historical, current

    #  ReAct: Razonamiento — BERTopic + cálculo Δ
    def _reason(
        self,
        historical_texts: list[dict],
        current_texts: list[dict],
    ) -> tuple[list[int], list[int], dict]:
        """
        BERTopic se entrena sobre el corpus completo (histórico + evaluación)
        para obtener tópicos más estables. El split temporal se aplica
        post-hoc sobre las asignaciones para el cálculo de Δ.

        Nota: BERTopic es no supervisado — entrenar sobre el corpus completo
        produce tópicos más robustos. El rigor del split temporal se preserva
        en _calculate_temporal_stats donde Δ compara días 1-60 vs días 61-90.

        Returns:
            (hist_topic_ids, curr_topic_ids, temporal_stats)
        """
        all_texts = historical_texts + current_texts
        all_docs = [t["text_for_topics"] for t in all_texts]

        logger.info(
            f"[Razonamiento] Ajustando BERTopic sobre {len(all_docs)} documentos "
            f"({len(historical_texts)} históricos + {len(current_texts)} evaluación)..."
        )
        all_topic_ids, _ = self._topic_model.fit_transform(all_docs)

        n_topics = len(set(all_topic_ids)) - (1 if -1 in all_topic_ids else 0)
        logger.info(f"[Razonamiento] {n_topics} tópicos detectados.")

        # Separar asignaciones por ventana temporal
        hist_topic_ids = list(all_topic_ids[:len(historical_texts)])
        curr_topic_ids = list(all_topic_ids[len(historical_texts):])

        logger.info(
            f"[Razonamiento] Split temporal — "
            f"Histórico: {sum(1 for t in hist_topic_ids if t != -1)} asignados | "
            f"Evaluación: {sum(1 for t in curr_topic_ids if t != -1)} asignados, "
            f"{sum(1 for t in curr_topic_ids if t == -1)} outliers."
        )

        temporal_stats = self._calculate_temporal_stats(
            historical_texts, current_texts,
            hist_topic_ids, curr_topic_ids,
        )

        return hist_topic_ids, curr_topic_ids, temporal_stats

    def _calculate_temporal_stats(
        self,
        historical_texts: list[dict],
        current_texts: list[dict],
        hist_topic_ids: list[int],
        curr_topic_ids: list[int],
    ) -> dict:
        """
        Calcula Δ usando el split temporal correcto:
          - Pesos históricos: calculados día a día sobre días 1-60 (baseline)
          - Peso actual: promedio sobre la ventana de evaluación días 61-90

        Returns:
            {topic_id: {daily_weights, current_weight, historical_mean, ...}}
        """
        from collections import defaultdict

        # Pesos diarios en ventana histórica
        hist_day_topics: dict[str, list[int]] = defaultdict(list)
        for t, tid in zip(historical_texts, hist_topic_ids):
            day = datetime.fromtimestamp(t["created_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
            hist_day_topics[day].append(tid)

        # Pesos diarios en ventana de evaluación
        curr_day_topics: dict[str, list[int]] = defaultdict(list)
        for t, tid in zip(current_texts, curr_topic_ids):
            day = datetime.fromtimestamp(t["created_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
            curr_day_topics[day].append(tid)

        # Tópicos válidos = los detectados en histórico (BERTopic.fit los define)
        unique_topics = set(t for t in hist_topic_ids if t != -1)
        total_texts = len(historical_texts) + len(current_texts)

        stats = {}
        for tid in unique_topics:
            # Peso del tópico por día en ventana histórica
            hist_daily: dict[str, float] = {}
            for day, tids in sorted(hist_day_topics.items()):
                n_day = len(tids)
                n_topic = sum(1 for t in tids if t == tid)
                hist_daily[day] = n_topic / n_day if n_day > 0 else 0.0

            # Peso del tópico por día en ventana de evaluación
            curr_daily: dict[str, float] = {}
            for day, tids in sorted(curr_day_topics.items()):
                n_day = len(tids)
                n_topic = sum(1 for t in tids if t == tid)
                curr_daily[day] = n_topic / n_day if n_day > 0 else 0.0

            hist_weights = list(hist_daily.values())
            historical_mean = float(np.mean(hist_weights)) if hist_weights else 0.0
            historical_std = float(np.std(hist_weights, ddof=0)) if hist_weights else 0.0

            current_weight = float(np.mean(list(curr_daily.values()))) if curr_daily else 0.0

            effective_std = max(historical_std, STD_FLOOR)
            delta = (current_weight - historical_mean) / effective_std

            n_current = sum(1 for t in curr_topic_ids if t == tid)
            n_historical = sum(1 for t in hist_topic_ids if t == tid)
            corpus_coverage = (n_current + n_historical) / total_texts

            # Días consecutivos creciendo en ventana de evaluación
            sorted_curr = sorted(curr_daily.items())
            consecutive_growth = 0
            if len(sorted_curr) >= 2:
                for i in range(1, len(sorted_curr)):
                    if sorted_curr[i][1] > sorted_curr[i - 1][1]:
                        consecutive_growth += 1
                    else:
                        consecutive_growth = 0

            # daily_weights combina ambas ventanas para visualización
            daily_weights = {**hist_daily, **curr_daily}

            stats[tid] = {
                "daily_weights": daily_weights,
                "current_weight": current_weight,
                "historical_mean": historical_mean,
                "historical_std": historical_std,
                "effective_std": effective_std,
                "delta": delta,
                "n_current": n_current,
                "n_historical": n_historical,
                "corpus_coverage": corpus_coverage,
                "consecutive_growth_days": consecutive_growth,
            }

            logger.debug(
                f"[Razonamiento] Tópico {tid}: Δ={delta:.2f} "
                f"(curr={current_weight:.4f}, mean={historical_mean:.4f}, "
                f"std={historical_std:.4f}→{effective_std:.4f}) "
                f"coverage={corpus_coverage:.3f}"
            )

        return stats

    #  ReAct: Acción — decisión de tendencia
    def _act(self, topic_id: int, stats: dict, ) -> tuple[str, str]:
        delta = stats["delta"]
        coverage = stats["corpus_coverage"]
        consec = stats["consecutive_growth_days"]
        n_current = stats["n_current"]
        curr_w = stats["current_weight"]
        hist_mean = stats["historical_mean"]
        if n_current < MIN_TOPIC_TEXTS:
            return "discarded", f"Insuficientes textos en ventana actual ({n_current} < {MIN_TOPIC_TEXTS})"
        if delta >= self.delta_high:
            if coverage > self.coverage_threshold:
                decision = "emerging_trend"
                reason = (
                    f"Δ={delta:.2f} ≥ {self.delta_high} y cobertura {coverage:.1%} > {self.coverage_threshold:.0%}. "
                    f"Peso actual={curr_w:.4f} vs media_hist={hist_mean:.4f}."
                )
            else:
                decision = "localized_spike"
                reason = (
                    f"Δ={delta:.2f} ≥ {self.delta_high} pero cobertura {coverage:.1%} ≤ {self.coverage_threshold:.0%}. "
                    f"Spike localizado, marcado para monitoreo."
                )
        elif delta >= self.delta_moderate:
            if consec >= 3:
                decision = "moderate_trend"
                reason = (
                    f"Δ={delta:.2f} en zona moderada [{self.delta_moderate},{self.delta_high}) "
                    f"y crecimiento {consec} días consecutivos."
                )
            elif curr_w < hist_mean:
                decision = "discarded"
                reason = (
                    f"Δ={delta:.2f} en zona moderada pero tópico decreciente "
                    f"(peso actual {curr_w:.4f} < media {hist_mean:.4f}). Pico pasajero."
                )
            else:
                decision = "discarded"
                reason = (
                    f"Δ={delta:.2f} en zona moderada pero sin {3} días consecutivos de crecimiento "
                    f"({consec} días)."
                )
        else:
            decision = "discarded"
            reason = f"Δ={delta:.2f} < {self.delta_moderate}. No es tendencia."

        logger.debug(f"[Acción] Tópico {topic_id}: {decision} — {reason}")
        return decision, reason

    #  ReAct: Registro
    def _record(
        self,
        historical_texts: list[dict],
        current_texts: list[dict],
        hist_topic_ids: list[int],
        curr_topic_ids: list[int],
        temporal_stats: dict,
        trend_decisions: dict,
    ) -> tuple[int, int]:
        """
        Persiste asignaciones de tópicos (histórico + evaluación) y resultados de tendencia.

        Returns:
            (n_assignments_inserted, n_trends_inserted)
        """
        now = datetime.now(tz=timezone.utc).isoformat()

        # Obtener mapa de etiquetas de BERTopic
        topic_info = self._topic_model.get_topic_info()
        topic_labels = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid != -1:
                topic_labels[tid] = row["Name"]  # e.g. "0_trump_tariff_trade"

        # Asignaciones: histórico (sin probs, solo topic_id)
        assignments = []
        for text_row, tid in zip(historical_texts, hist_topic_ids):
            assignments.append({
                "preprocessed_text_id": text_row["id"],
                "source_id": text_row["source_id"],
                "source_type": text_row["source_type"],
                "subreddit": text_row["subreddit"],
                "created_utc": text_row["created_utc"],
                "topic_id": int(tid),
                "topic_label": topic_labels.get(int(tid)),
                "topic_probability": None,
                "model_run_id": self.model_run_id,
                "assigned_at": now,
            })

        # Asignaciones: evaluación
        for text_row, tid in zip(current_texts, curr_topic_ids):
            assignments.append({
                "preprocessed_text_id": text_row["id"],
                "source_id": text_row["source_id"],
                "source_type": text_row["source_type"],
                "subreddit": text_row["subreddit"],
                "created_utc": text_row["created_utc"],
                "topic_id": int(tid),
                "topic_label": topic_labels.get(int(tid)),
                "topic_probability": None,
                "model_run_id": self.model_run_id,
                "assigned_at": now,
            })

        n_assigned = self.db.insert_topic_assignments_batch(assignments)
        logger.info(f"[Registro] {n_assigned} asignaciones de tópicos guardadas.")

        # Tendencias
        min_dt = datetime.fromtimestamp(self._min_ts, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(self._max_ts, tz=timezone.utc)
        cutoff_dt = datetime.fromtimestamp(self._current_cutoff, tz=timezone.utc)

        trends = []
        for tid, (decision, reason) in trend_decisions.items():
            s = temporal_stats[tid]
            trends.append({
                "model_run_id": self.model_run_id,
                "topic_id": int(tid),
                "topic_label": topic_labels.get(int(tid)),
                "current_weight": s["current_weight"],
                "historical_mean": s["historical_mean"],
                "historical_std": s["historical_std"],
                "effective_std": s["effective_std"],
                "delta": s["delta"],
                "current_window_start": cutoff_dt.date().isoformat(),
                "current_window_end": max_dt.date().isoformat(),
                "historical_window_start": min_dt.date().isoformat(),
                "historical_window_end": cutoff_dt.date().isoformat(),
                "n_current_texts": s["n_current"],
                "n_historical_texts": s["n_historical"],
                "corpus_coverage": s["corpus_coverage"],
                "consecutive_growth_days": s["consecutive_growth_days"],
                "trend_decision": decision,
                "trend_reason": reason,
                "daily_weights_json": json.dumps(s["daily_weights"]),
                "analyzed_at": now,
            })

        n_trends = self.db.insert_trend_analysis_batch(trends)
        logger.info(f"[Registro] {n_trends} análisis de tendencia guardados.")

        return n_assigned, n_trends

    #  Ciclo principal
    def run(self, limit: int = 50000) -> dict:
        """
        Ejecuta el ciclo ReAct completo.
        Args:
            limit: Máximo de textos a cargar para el modelo.
        Returns:
            dict con métricas del ciclo.
        """
        self._load_models()

        # Observación
        historical_texts, current_texts = self._observe(limit)
        if not historical_texts or not current_texts:
            logger.info("No hay suficientes textos en ambas ventanas para analizar tendencias.")
            return {"total_texts": 0}

        total = len(historical_texts) + len(current_texts)

        # Razonamiento
        hist_topic_ids, curr_topic_ids, temporal_stats = self._reason(
            historical_texts, current_texts
        )

        n_topics = len(temporal_stats)
        n_outliers = sum(1 for t in curr_topic_ids if t == -1)

        # Acción
        trend_decisions = {}
        decision_counts = {
            "emerging_trend": 0,
            "localized_spike": 0,
            "moderate_trend": 0,
            "discarded": 0,
        }

        for tid, stats in temporal_stats.items():
            decision, reason = self._act(tid, stats)
            trend_decisions[tid] = (decision, reason)
            decision_counts[decision] += 1

        # Tópicos con tendencia real
        trending = [
            tid for tid, (dec, _) in trend_decisions.items()
            if dec in ("emerging_trend", "localized_spike", "moderate_trend")
        ]

        # Registro
        n_assigned, n_trends = self._record(
            historical_texts, current_texts,
            hist_topic_ids, curr_topic_ids,
            temporal_stats, trend_decisions,
        )

        # Resumen con tópicos top por Δ
        top_trends = sorted(
            [
                {
                    "topic_id": tid,
                    "label": self._topic_model.get_topic_info()
                    .set_index("Topic")
                    .loc[tid, "Name"]
                    if tid in self._topic_model.get_topic_info()["Topic"].values
                    else str(tid),
                    "delta": temporal_stats[tid]["delta"],
                    "decision": trend_decisions[tid][0],
                    "coverage": temporal_stats[tid]["corpus_coverage"],
                }
                for tid in temporal_stats
            ],
            key=lambda x: x["delta"],
            reverse=True,
        )[:15]

        summary = {
            "model_run_id": self.model_run_id,
            "total_texts": total,
            "n_outliers": n_outliers,
            "n_topics_detected": n_topics,
            "n_assignments_saved": n_assigned,
            "trend_decisions": decision_counts,
            "trending_topics_count": len(trending),
            "top_topics_by_delta": top_trends,
        }

        logger.info("=" * 60)
        logger.info("RESUMEN AGENTE DE TENDENCIAS")
        logger.info(f"  Run ID          : {self.model_run_id}")
        logger.info(f"  Textos procesados: {total}")
        logger.info(f"  Tópicos detectados: {n_topics} (+ {n_outliers} outliers)")
        logger.info(f"  Decisiones      : {decision_counts}")
        logger.info(f"  Tendencias activas: {len(trending)}")
        logger.info("-" * 60)
        logger.info("  TOP TÓPICOS POR Δ:")
        for t in top_trends[:8]:
            logger.info(
                f"    [{t['decision'][:4].upper()}] Δ={t['delta']:+.2f} "
                f"cov={t['coverage']:.1%}  {t['label']}"
            )
        logger.info("=" * 60)

        return summary
