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

Ventana adaptativa:
    W_eval = max(W_stat, W_lifecycle)

    W_stat = N_min / λ
      Garantiza estabilidad estadística. N_min se deriva del Teorema
      Central del Límite para proporciones binomiales: para estimar la
      proporción p de un tópico con margen de error δ = p/2 al 95% de
      confianza, se necesitan N_min = z² × p(1-p) / δ² observaciones
      (Cochran, 1977). Con p_min = 5% (COVERAGE_THRESHOLD):
      N_min = 1.96² × 0.05 × 0.95 / 0.025² ≈ 292 textos.

    W_lifecycle = α × T_½
      Garantiza capturar el ciclo de vida del tópico. T_½ es la vida
      media empírica, medida ajustando decaimiento exponencial f(t) =
      e^(-λt) a la curva post-pico de cada tópico (Leskovec et al.,
      2009). α = 2 captura ~75% del ciclo (suficiente para detectar
      la fase de crecimiento + pico).

    Con corpus denso (alto λ), W_lifecycle domina → ventana corta.
    Con corpus escaso (bajo λ), W_stat domina → ventana más larga.

    Refs:
      - Cochran (1977): Sampling Techniques — fórmula de tamaño muestral
      - Leskovec et al. (2009): Meme-tracking — decaimiento exponencial
      - Kleinberg (2002): Bursty Structure in Streams — ventanas adaptativas
      - Mathioudakis & Koudas (2010): TwitterMonitor — detección en streams

Decisiones (umbrales calibrados con corpus real):
    Δ ≥ 1.5 y coverage > 5%  → emerging_trend
    Δ ≥ 1.5 y coverage ≤ 5%  → localized_spike
    1.0 ≤ Δ < 1.5 y peso actual > media → moderate_trend
    1.0 ≤ Δ < 1.5 y peso actual < media → discarded (pico pasajero)
    Δ < 1.0                            → discarded
"""

import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger
from scipy.optimize import curve_fit

from src.database.db_manager import DatabaseManager

#  Umbrales y configuración
DELTA_HIGH = 1.5           # Umbral para tendencia/spike (calibrado con corpus real de 89 días)
DELTA_MODERATE = 1.0       # Umbral para tendencia moderada
COVERAGE_THRESHOLD = 0.05  # 5% del corpus = tendencia emergente vs spike localizado
STD_FLOOR = 0.005          # Floor mínimo para desv. estándar (evita Δ artificialmente alto)
MIN_TOPIC_TEXTS = 10       # Textos mínimos en ventana actual para evaluar un tópico

# Parámetros de ventana adaptativa
CONFIDENCE_Z = 1.96        # z para intervalo de confianza 95%
LIFECYCLE_ALPHA = 2        # Multiplicador de vida media (α=2 → captura ~75% del ciclo)
FALLBACK_WINDOW_DAYS = 2   # Fallback si no se puede calcular half-life
MIN_WINDOW_HOURS = 6       # Mínimo absoluto de ventana (evita ruido extremo)
MAX_WINDOW_DAYS = 7        # Máximo absoluto (evita diluir señal)
MIN_PEAK_TEXTS = 20        # Mínimo de textos en pico para calcular half-life de un tópico

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
        current_days: Optional[float] = None,  # None = cálculo adaptativo
        delta_high: float = DELTA_HIGH,
        delta_moderate: float = DELTA_MODERATE,
        coverage_threshold: float = COVERAGE_THRESHOLD,
    ):
        self.db = db or DatabaseManager()
        self.n_topics = n_topics
        self._forced_window = current_days  # None = adaptativo, float = forzado
        self.current_days = current_days or FALLBACK_WINDOW_DAYS
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

    #  Ventana adaptativa

    def _calculate_adaptive_window(self, texts: list[dict]) -> float:
        """
        Calcula la ventana de evaluación óptima basada en dos criterios:

        W_eval = max(W_stat, W_lifecycle)

        1. W_stat = N_min / λ  (estabilidad estadística)
           N_min = z² × p(1-p) / δ²  donde p = COVERAGE_THRESHOLD, δ = p/2
           Cochran (1977): tamaño muestral para proporciones binomiales.

        2. W_lifecycle = α × T_½  (ciclo de vida empírico)
           T_½ = mediana de vida media de tópicos con buen ajuste exponencial.
           Leskovec et al. (2009): decaimiento exponencial de cascadas.

        Returns:
            Ventana óptima en días (float).
        """
        timestamps = [t["created_utc"] for t in texts if t["created_utc"]]
        if not timestamps:
            return FALLBACK_WINDOW_DAYS

        max_ts = max(timestamps)
        min_ts = min(timestamps)
        total_days = (max_ts - min_ts) / 86400
        n_docs = len(timestamps)

        if total_days <= 0:
            return FALLBACK_WINDOW_DAYS

        # λ = tasa de documentos por día
        lam = n_docs / total_days

        # W_stat: ventana mínima para estabilidad estadística
        # N_min = z² × p(1-p) / δ²  con p = coverage_threshold, δ = p/2
        p = self.coverage_threshold
        delta_margin = p / 2
        n_min = (CONFIDENCE_Z ** 2 * p * (1 - p)) / (delta_margin ** 2)
        w_stat_days = n_min / lam

        logger.info(
            f"[Ventana adaptativa] W_stat: N_min={n_min:.0f} textos, "
            f"λ={lam:.0f} docs/día → W_stat={w_stat_days:.2f} días "
            f"({w_stat_days * 24:.1f} horas)"
        )

        # W_lifecycle: basado en vida media empírica de tópicos
        half_life_hours = self._estimate_corpus_half_life(texts)

        if half_life_hours is not None:
            w_life_days = (LIFECYCLE_ALPHA * half_life_hours) / 24
            logger.info(
                f"[Ventana adaptativa] W_lifecycle: T_½={half_life_hours:.1f}h, "
                f"α={LIFECYCLE_ALPHA} → W_lifecycle={w_life_days:.2f} días "
                f"({w_life_days * 24:.1f} horas)"
            )
        else:
            w_life_days = FALLBACK_WINDOW_DAYS
            logger.info(
                f"[Ventana adaptativa] No se pudo calcular T_½, "
                f"usando fallback W_lifecycle={FALLBACK_WINDOW_DAYS} días"
            )

        # W_eval = max(W_stat, W_lifecycle), acotado por [MIN, MAX]
        w_eval = max(w_stat_days, w_life_days)
        w_eval = max(w_eval, MIN_WINDOW_HOURS / 24)
        w_eval = min(w_eval, MAX_WINDOW_DAYS)

        logger.info(
            f"[Ventana adaptativa] W_eval = max({w_stat_days:.2f}, {w_life_days:.2f}) "
            f"= {w_eval:.2f} días ({w_eval * 24:.1f} horas)"
        )

        return round(w_eval, 2)

    def _estimate_corpus_half_life(self, texts: list[dict]) -> Optional[float]:
        """
        Estima la vida media (T_½) mediana de los tópicos del corpus.

        Usa un modelo BERTopic rápido (solo para medir T_½, no el modelo final)
        o datos históricos de topic_assignments si existen.

        Método: para cada tópico con pico ≥ MIN_PEAK_TEXTS, ajusta
        f(t) = e^(-λt) a la curva post-pico. T_½ = ln(2)/λ.

        Returns:
            Mediana de T_½ en horas, o None si no hay suficientes datos.
        """
        # Intentar usar topic_assignments existentes en la BD
        try:
            import sqlite3
            db_path = self.db.db_path if hasattr(self.db, 'db_path') else None
            if db_path:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM topic_assignments WHERE topic_id != -1")
                n_assignments = cur.fetchone()[0]

                if n_assignments > 1000:
                    return self._half_life_from_db(conn)
                conn.close()
        except Exception as e:
            logger.debug(f"No se pudieron usar topic_assignments existentes: {e}")

        # Fallback: calcular desde los timestamps del corpus sin tópicos
        return self._half_life_from_timestamps(texts)

    def _half_life_from_db(self, conn) -> Optional[float]:
        """Calcula T_½ desde topic_assignments existentes en la BD."""
        cur = conn.cursor()
        cur.execute("""
            SELECT topic_id, created_utc
            FROM topic_assignments
            WHERE topic_id != -1
        """)

        topic_days: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for topic_id, created_utc in cur.fetchall():
            day = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime("%Y-%m-%d")
            topic_days[topic_id][day] += 1
        conn.close()

        half_lives = []
        for tid, daily in topic_days.items():
            hl = self._fit_half_life(daily)
            if hl is not None:
                half_lives.append(hl)

        if len(half_lives) < 5:
            logger.info(f"[Half-life] Solo {len(half_lives)} tópicos con ajuste válido, insuficiente.")
            return None

        median_hl = float(np.median(half_lives))
        logger.info(
            f"[Half-life] {len(half_lives)} tópicos analizados desde BD. "
            f"Mediana T_½ = {median_hl:.1f}h (P25={np.percentile(half_lives, 25):.1f}h, "
            f"P75={np.percentile(half_lives, 75):.1f}h)"
        )
        return median_hl

    def _half_life_from_timestamps(self, texts: list[dict]) -> Optional[float]:
        """
        Estimación gruesa de T_½ sin tópicos: usa la autocorrelación
        temporal del volumen de discusión. Menos preciso pero funciona
        como fallback cuando no hay topic_assignments.
        """
        day_counts: dict[str, int] = defaultdict(int)
        for t in texts:
            if t.get("created_utc"):
                day = datetime.fromtimestamp(t["created_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
                day_counts[day] += 1

        if len(day_counts) < 7:
            return None

        sorted_days = sorted(day_counts.items())
        counts = np.array([c for _, c in sorted_days], dtype=float)

        # Autocorrelación normalizada — el lag donde cae a 0.5 es ~T_½
        mean_c = np.mean(counts)
        var_c = np.var(counts)
        if var_c == 0:
            return None

        max_lag = min(len(counts) // 2, 14)
        for lag in range(1, max_lag):
            autocorr = np.mean((counts[:-lag] - mean_c) * (counts[lag:] - mean_c)) / var_c
            if autocorr < 0.5:
                half_life_hours = lag * 24  # cada lag = 1 día
                logger.info(f"[Half-life fallback] Autocorrelación < 0.5 en lag={lag} días → T_½ ≈ {half_life_hours}h")
                return half_life_hours

        return None

    def _fit_half_life(self, daily_counts: dict[str, int]) -> Optional[float]:
        """
        Ajusta decaimiento exponencial f(t) = e^(-λt) a la curva post-pico
        de un tópico. Retorna T_½ = ln(2)/λ en horas si R² > 0.5.
        """
        sorted_days = sorted(daily_counts.items())
        counts = np.array([c for _, c in sorted_days])

        peak_idx = np.argmax(counts)
        peak_count = counts[peak_idx]

        if peak_count < MIN_PEAK_TEXTS:
            return None

        post_peak = counts[peak_idx:]
        if len(post_peak) < 3:
            return None

        normalized = post_peak / peak_count
        x = np.arange(len(normalized))

        try:
            def exp_decay(t, lam):
                return np.exp(-lam * t)

            popt, _ = curve_fit(exp_decay, x, normalized, p0=[0.5], maxfev=5000)
            lam = popt[0]
            if lam <= 0:
                return None

            predicted = exp_decay(x, lam)
            ss_res = np.sum((normalized - predicted) ** 2)
            ss_tot = np.sum((normalized - np.mean(normalized)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r_squared < 0.5:
                return None

            return (np.log(2) / lam) * 24  # días → horas
        except (RuntimeError, ValueError):
            return None

    #  ReAct: Observación
    def _observe(self, limit: int) -> tuple[list[dict], list[dict]]:
        """
        Carga textos, calcula la ventana adaptativa, y aplica el split temporal.

        La ventana de evaluación se determina como:
          W_eval = max(W_stat, W_lifecycle)
        a menos que el usuario haya forzado un valor en el constructor.

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

        # Calcular ventana adaptativa (o usar la forzada)
        if self._forced_window is not None:
            self.current_days = self._forced_window
            logger.info(f"[Observación] Ventana forzada por usuario: {self.current_days} días")
        else:
            self.current_days = self._calculate_adaptive_window(texts)

        # Split: ventana actual = últimos current_days desde el máximo
        self._current_cutoff = self._max_ts - (self.current_days * 86400)

        historical = [t for t in texts if t["created_utc"] < self._current_cutoff]
        current = [t for t in texts if t["created_utc"] >= self._current_cutoff]

        max_date = datetime.fromtimestamp(self._max_ts, tz=timezone.utc).date()
        min_date = datetime.fromtimestamp(self._min_ts, tz=timezone.utc).date()
        cutoff_date = datetime.fromtimestamp(self._current_cutoff, tz=timezone.utc).date()
        total_days = (self._max_ts - self._min_ts) / 86400

        logger.info(
            f"[Observación] Split temporal (ventana={self.current_days:.2f} días) — "
            f"Histórico: {min_date} → {cutoff_date} ({len(historical)} textos, "
            f"~{total_days - self.current_days:.0f} días) | "
            f"Evaluación: {cutoff_date} → {max_date} ({len(current)} textos, "
            f"~{self.current_days:.1f} días)"
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
            if curr_w < hist_mean:
                decision = "discarded"
                reason = (
                    f"Δ={delta:.2f} en zona moderada pero tópico decreciente "
                    f"(peso actual {curr_w:.4f} < media {hist_mean:.4f}). Pico pasajero."
                )
            else:
                decision = "moderate_trend"
                reason = (
                    f"Δ={delta:.2f} en zona moderada [{self.delta_moderate},{self.delta_high}) "
                    f"con peso actual {curr_w:.4f} > media histórica {hist_mean:.4f}."
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
            "evaluation_window_days": self.current_days,
            "top_topics_by_delta": top_trends,
        }

        logger.info("=" * 60)
        logger.info("RESUMEN AGENTE DE TENDENCIAS")
        logger.info(f"  Run ID          : {self.model_run_id}")
        logger.info(f"  Ventana eval    : {self.current_days:.2f} días ({self.current_days * 24:.1f} horas)")
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
