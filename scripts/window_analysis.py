"""
Análisis empírico de ventana temporal óptima para detección de tendencias.

Metodología:
1. Half-life analysis: calcula cuánto tiempo tarda un tópico en perder
   50% de su actividad pico (vida media). Esto indica la duración natural
   del discurso político en r/politics.
2. Sensitivity analysis: ejecuta el cálculo de Δ con ventanas de 1-7 días
   y compara cuántas tendencias detecta cada una y su estabilidad.

Basado en literatura:
- Murdock et al. (CMU): discusiones políticas en Reddit tienen ciclo de vida
  de 22-30 horas.
- Leskovec et al.: memes informativos siguen curva exponencial de decaimiento.
- Grootendorst (BERTopic): recomienda <50 bins temporales para estabilidad.
"""

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit


DB_PATH = "data/reddit_political.db"


def load_topic_daily_activity(conn):
    """
    Carga actividad diaria de cada tópico desde topic_assignments.
    Returns: {topic_id: {date_str: count}}
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT topic_id, created_utc
        FROM topic_assignments
        WHERE topic_id != -1
    """)

    topic_days = defaultdict(lambda: defaultdict(int))
    for topic_id, created_utc in cur.fetchall():
        day = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime("%Y-%m-%d")
        topic_days[topic_id][day] += 1

    return dict(topic_days)


def calculate_half_life(daily_counts: dict[str, int], min_peak: int = 20) -> dict | None:
    """
    Calcula la vida media de un tópico basado en su curva de decaimiento
    después del pico de actividad.

    Args:
        daily_counts: {date_str: count}
        min_peak: mínimo de textos en el pico para considerar el tópico

    Returns:
        dict con half_life_hours, peak_date, peak_count, r_squared
        o None si no se puede calcular
    """
    if not daily_counts:
        return None

    sorted_days = sorted(daily_counts.items())
    dates = [d for d, _ in sorted_days]
    counts = np.array([c for _, c in sorted_days])

    # Encontrar el pico
    peak_idx = np.argmax(counts)
    peak_count = counts[peak_idx]

    if peak_count < min_peak:
        return None

    # Tomar la curva post-pico (decaimiento)
    post_peak = counts[peak_idx:]
    if len(post_peak) < 3:  # necesitamos al menos 3 puntos para ajustar
        return None

    # Normalizar respecto al pico
    normalized = post_peak / peak_count
    x = np.arange(len(normalized))  # días desde el pico

    # Ajustar exponencial decreciente: f(t) = exp(-λt)
    try:
        def exp_decay(t, lam):
            return np.exp(-lam * t)

        popt, _ = curve_fit(exp_decay, x, normalized, p0=[0.5], maxfev=5000)
        lam = popt[0]

        if lam <= 0:
            return None

        half_life_days = np.log(2) / lam
        half_life_hours = half_life_days * 24

        # R² del ajuste
        predicted = exp_decay(x, lam)
        ss_res = np.sum((normalized - predicted) ** 2)
        ss_tot = np.sum((normalized - np.mean(normalized)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "half_life_hours": round(half_life_hours, 1),
            "half_life_days": round(half_life_days, 2),
            "peak_date": dates[peak_idx],
            "peak_count": int(peak_count),
            "decay_rate_lambda": round(lam, 4),
            "r_squared": round(r_squared, 4),
            "n_post_peak_days": len(post_peak),
        }
    except (RuntimeError, ValueError):
        return None


def sensitivity_analysis(conn, windows=[1, 2, 3, 5, 7]):
    """
    Ejecuta el cálculo de Δ con diferentes tamaños de ventana
    y compara los resultados.
    """
    cur = conn.cursor()

    # Cargar todos los textos con timestamps y topic_ids
    cur.execute("""
        SELECT topic_id, created_utc
        FROM topic_assignments
        WHERE topic_id != -1
    """)
    all_data = cur.fetchall()

    timestamps = [row[1] for row in all_data]
    max_ts = max(timestamps)
    min_ts = min(timestamps)

    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE SENSIBILIDAD DE VENTANA TEMPORAL")
    print(f"{'='*70}")
    print(f"Corpus: {datetime.fromtimestamp(min_ts, tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(max_ts, tz=timezone.utc).date()}")
    print(f"Total asignaciones: {len(all_data):,}")

    DELTA_HIGH = 1.5
    DELTA_MODERATE = 1.0
    COVERAGE_THRESHOLD = 0.05
    STD_FLOOR = 0.005
    MIN_TOPIC_TEXTS = 10

    results = {}

    for window_days in windows:
        cutoff = max_ts - (window_days * 86400)

        # Separar histórico vs actual
        hist_topics = defaultdict(lambda: defaultdict(int))
        curr_topics = defaultdict(lambda: defaultdict(int))
        hist_total_by_day = defaultdict(int)
        curr_total_by_day = defaultdict(int)

        n_hist = 0
        n_curr = 0

        for topic_id, ts in all_data:
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            if ts < cutoff:
                hist_topics[topic_id][day] += 1
                hist_total_by_day[day] += 1
                n_hist += 1
            else:
                curr_topics[topic_id][day] += 1
                curr_total_by_day[day] += 1
                n_curr += 1

        # Calcular Δ para cada tópico
        all_topic_ids = set(list(hist_topics.keys()) + list(curr_topics.keys()))

        decisions = {"emerging_trend": 0, "localized_spike": 0,
                     "moderate_trend": 0, "discarded": 0}
        trending_deltas = []

        for tid in all_topic_ids:
            # Pesos diarios históricos
            hist_weights = []
            for day in sorted(hist_total_by_day.keys()):
                total_day = hist_total_by_day[day]
                topic_day = hist_topics[tid].get(day, 0)
                hist_weights.append(topic_day / total_day if total_day > 0 else 0)

            # Pesos diarios actuales
            curr_weights = []
            for day in sorted(curr_total_by_day.keys()):
                total_day = curr_total_by_day[day]
                topic_day = curr_topics[tid].get(day, 0)
                curr_weights.append(topic_day / total_day if total_day > 0 else 0)

            if not hist_weights or not curr_weights:
                continue

            hist_mean = np.mean(hist_weights)
            hist_std = np.std(hist_weights, ddof=0)
            curr_weight = np.mean(curr_weights)
            effective_std = max(hist_std, STD_FLOOR)
            delta = (curr_weight - hist_mean) / effective_std

            n_current = sum(curr_topics[tid].values())
            n_total = n_current + sum(hist_topics[tid].values())
            coverage = n_total / (n_hist + n_curr)

            if n_current < MIN_TOPIC_TEXTS:
                decisions["discarded"] += 1
            elif delta >= DELTA_HIGH and coverage > COVERAGE_THRESHOLD:
                decisions["emerging_trend"] += 1
                trending_deltas.append(delta)
            elif delta >= DELTA_HIGH:
                decisions["localized_spike"] += 1
                trending_deltas.append(delta)
            elif delta >= DELTA_MODERATE and curr_weight >= hist_mean:
                decisions["moderate_trend"] += 1
                trending_deltas.append(delta)
            else:
                decisions["discarded"] += 1

        n_trending = decisions["emerging_trend"] + decisions["localized_spike"] + decisions["moderate_trend"]
        avg_delta = np.mean(trending_deltas) if trending_deltas else 0

        results[window_days] = {
            "n_current_texts": n_curr,
            "n_historical_texts": n_hist,
            "n_current_days": len(curr_total_by_day),
            "decisions": decisions,
            "n_trending": n_trending,
            "avg_trending_delta": round(avg_delta, 2),
            "signal_ratio": f"{decisions['emerging_trend']}/{n_trending}" if n_trending > 0 else "0/0",
        }

        print(f"\n--- Ventana = {window_days} días ---")
        print(f"  Textos actual: {n_curr:,} | Días actual: {len(curr_total_by_day)}")
        print(f"  Emerging: {decisions['emerging_trend']} | Spikes: {decisions['localized_spike']} | "
              f"Moderate: {decisions['moderate_trend']} | Discarded: {decisions['discarded']}")
        print(f"  Total trending: {n_trending} | Avg Δ trending: {avg_delta:.2f}")

    return results


def main():
    conn = sqlite3.connect(DB_PATH)

    # ================================================================
    # 1. HALF-LIFE ANALYSIS
    # ================================================================
    print("="*70)
    print("1. ANÁLISIS DE VIDA MEDIA (HALF-LIFE) DE TÓPICOS")
    print("="*70)
    print("Midiendo cuánto tiempo tarda un tópico en perder 50% de su actividad pico.\n")

    topic_activity = load_topic_daily_activity(conn)
    print(f"Tópicos totales: {len(topic_activity)}")

    half_lives = []
    good_fits = []  # R² > 0.5

    for tid, daily in topic_activity.items():
        result = calculate_half_life(daily, min_peak=20)
        if result:
            half_lives.append(result)
            if result["r_squared"] > 0.5:
                good_fits.append(result)

    print(f"Tópicos con suficiente actividad para análisis: {len(half_lives)}")
    print(f"Tópicos con buen ajuste exponencial (R² > 0.5): {len(good_fits)}")

    if good_fits:
        hl_hours = [r["half_life_hours"] for r in good_fits]
        hl_days = [r["half_life_days"] for r in good_fits]

        print(f"\n--- Estadísticas de vida media (R² > 0.5, n={len(good_fits)}) ---")
        print(f"  Media:    {np.mean(hl_hours):.1f} horas ({np.mean(hl_days):.2f} días)")
        print(f"  Mediana:  {np.median(hl_hours):.1f} horas ({np.median(hl_days):.2f} días)")
        print(f"  Std:      {np.std(hl_hours):.1f} horas")
        print(f"  P25:      {np.percentile(hl_hours, 25):.1f} horas")
        print(f"  P75:      {np.percentile(hl_hours, 75):.1f} horas")
        print(f"  Min:      {min(hl_hours):.1f} horas")
        print(f"  Max:      {max(hl_hours):.1f} horas")

        # Distribución por rangos
        ranges = [(0, 24), (24, 48), (48, 72), (72, 120), (120, 168), (168, float('inf'))]
        labels = ["< 1 día", "1-2 días", "2-3 días", "3-5 días", "5-7 días", "> 7 días"]
        print(f"\n--- Distribución de vida media ---")
        for (lo, hi), label in zip(ranges, labels):
            count = sum(1 for h in hl_hours if lo <= h < hi)
            pct = count / len(hl_hours) * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:>10}: {count:3d} ({pct:5.1f}%) {bar}")

        # Top 10 tópicos con vida media más corta (los más "explosivos")
        print(f"\n--- Top 10 tópicos con vida media más corta ---")
        sorted_fits = sorted(good_fits, key=lambda x: x["half_life_hours"])
        for i, r in enumerate(sorted_fits[:10]):
            print(f"  {i+1}. {r['half_life_hours']:6.1f}h (R²={r['r_squared']:.3f}) "
                  f"peak={r['peak_count']} @ {r['peak_date']}")

    if half_lives:
        all_hl_hours = [r["half_life_hours"] for r in half_lives]
        print(f"\n--- Estadísticas incluyendo todos los ajustes (n={len(half_lives)}) ---")
        print(f"  Media:    {np.mean(all_hl_hours):.1f} horas ({np.mean(all_hl_hours)/24:.2f} días)")
        print(f"  Mediana:  {np.median(all_hl_hours):.1f} horas ({np.median(all_hl_hours)/24:.2f} días)")

    # ================================================================
    # 2. SENSITIVITY ANALYSIS
    # ================================================================
    sensitivity_results = sensitivity_analysis(conn, windows=[1, 2, 3, 5, 7])

    # ================================================================
    # 3. RECOMENDACIÓN
    # ================================================================
    print(f"\n{'='*70}")
    print("3. RECOMENDACIÓN")
    print(f"{'='*70}")

    if good_fits:
        median_hl = np.median(hl_hours)
        recommended_hours = median_hl * 3  # 3x half-life captura ~87.5% del ciclo
        recommended_days = recommended_hours / 24

        print(f"\nVida media mediana: {median_hl:.1f} horas")
        print(f"Ventana recomendada (3× vida media): {recommended_hours:.0f} horas "
              f"≈ {recommended_days:.1f} días")
        print(f"\nJustificación:")
        print(f"  - 1× vida media captura ~50% del ciclo de un tópico")
        print(f"  - 2× vida media captura ~75% del ciclo")
        print(f"  - 3× vida media captura ~87.5% del ciclo (recomendado)")
        print(f"\nReferencia literatura:")
        print(f"  - Murdock et al. (CMU): ciclo político Reddit = 22-30h")
        print(f"  - Leskovec et al.: decaimiento exponencial de memes informativos")
        print(f"  - Dato empírico de este corpus: mediana = {median_hl:.1f}h")

    # Tabla resumen de sensibilidad
    print(f"\n--- Tabla resumen sensibilidad ---")
    print(f"{'Ventana':>8} | {'Textos':>8} | {'Emerging':>8} | {'Spikes':>8} | {'Moderate':>8} | {'Total':>6} | {'Avg Δ':>6}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
    for w, r in sorted(sensitivity_results.items()):
        d = r["decisions"]
        print(f"{w:>7}d | {r['n_current_texts']:>8,} | {d['emerging_trend']:>8} | "
              f"{d['localized_spike']:>8} | {d['moderate_trend']:>8} | {r['n_trending']:>6} | {r['avg_trending_delta']:>6.2f}")

    conn.close()


if __name__ == "__main__":
    main()
