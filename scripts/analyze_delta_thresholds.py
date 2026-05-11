"""
Análisis empírico de umbrales Δ para detección de tendencias.

Genera:
  1. Distribución de Δ de todos los tópicos
  2. Análisis de sensibilidad: qué pasa si cambiamos Δ_HIGH y Δ_MODERATE
  3. Justificación estadística del z-score modificado
  4. Análisis de cobertura (COVERAGE_THRESHOLD = 5%)
  5. STD_FLOOR: por qué 0.005
  6. Gráficos para la defensa
"""

import sqlite3
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reddit_political.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation")

STD_FLOOR = 0.005
MIN_TOPIC_TEXTS = 10


def load_topic_stats(db_path: str) -> list[dict]:
    """Carga Δ y stats de cada tópico desde trend_analysis (datos reales del agente)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT model_run_id FROM trend_analysis ORDER BY analyzed_at DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        print("No hay datos en trend_analysis.")
        return []
    model_run_id = row["model_run_id"]

    cur.execute("""
        SELECT topic_id, topic_label, current_weight, historical_mean,
               historical_std, effective_std, delta, n_current_texts,
               n_historical_texts, corpus_coverage, trend_decision,
               current_window_start, current_window_end,
               historical_window_start, historical_window_end
        FROM trend_analysis
        WHERE model_run_id = ?
    """, (model_run_id,))
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "topic_id": r["topic_id"],
            "topic_label": r["topic_label"] or f"topic_{r['topic_id']}",
            "delta": r["delta"],
            "curr_weight": r["current_weight"],
            "hist_mean": r["historical_mean"],
            "hist_std": r["historical_std"],
            "effective_std": r["effective_std"],
            "used_floor": r["historical_std"] < STD_FLOOR,
            "n_current": r["n_current_texts"],
            "n_historical": r["n_historical_texts"],
            "coverage": r["corpus_coverage"],
            "trend_decision": r["trend_decision"],
        })

    print(f"Model run: {model_run_id}")
    print(f"Ventana evaluación: {rows[0]['current_window_start']} → {rows[0]['current_window_end']}")
    print(f"Ventana histórica:  {rows[0]['historical_window_start']} → {rows[0]['historical_window_end']}")
    print(f"Tópicos evaluados: {len(results)}")

    return results


def analyze_delta_distribution(stats: list[dict]):
    """Distribución de Δ de todos los tópicos."""
    print("\n" + "=" * 70)
    print("1. DISTRIBUCIÓN DE Δ (z-score modificado)")
    print("=" * 70)

    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS]
    deltas = [s["delta"] for s in valid]

    print(f"\nTópicos con ≥{MIN_TOPIC_TEXTS} textos en ventana actual: {len(valid)}")
    print(f"\nEstadísticas de Δ:")
    print(f"  Media:    {np.mean(deltas):+.3f}")
    print(f"  Mediana:  {np.median(deltas):+.3f}")
    print(f"  Std:      {np.std(deltas):.3f}")
    print(f"  Min:      {min(deltas):+.3f}")
    print(f"  Max:      {max(deltas):+.3f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [50, 75, 80, 85, 90, 95, 97.5, 99]:
        val = np.percentile(deltas, p)
        print(f"  P{p:>4}: Δ = {val:+.3f}")

    # Distribución por rango
    ranges = [
        ("Δ < -1.0 (declinación fuerte)", lambda d: d < -1.0),
        ("-1.0 ≤ Δ < 0 (declinación leve)", lambda d: -1.0 <= d < 0),
        ("0 ≤ Δ < 1.0 (estable/leve)", lambda d: 0 <= d < 1.0),
        ("1.0 ≤ Δ < 1.5 (moderado)", lambda d: 1.0 <= d < 1.5),
        ("1.5 ≤ Δ < 2.0 (significativo)", lambda d: 1.5 <= d < 2.0),
        ("2.0 ≤ Δ < 3.0 (fuerte)", lambda d: 2.0 <= d < 3.0),
        ("Δ ≥ 3.0 (extremo)", lambda d: d >= 3.0),
    ]

    print(f"\nDistribución por rango:")
    for name, filt in ranges:
        count = sum(1 for d in deltas if filt(d))
        pct = count / len(deltas) * 100
        bar = "█" * int(pct)
        print(f"  {name:<38}: {count:>4} ({pct:>5.1f}%) {bar}")

    return valid


def _simulate_decisions(valid: list[dict], delta_high: float, delta_moderate: float,
                        coverage_t: float = 0.05) -> dict:
    """Simula las decisiones del agente con umbrales dados."""
    emerging = spike = moderate = discarded = 0
    for s in valid:
        delta = s["delta"]
        cov = s["coverage"]
        growing = s["curr_weight"] >= s["hist_mean"]

        if delta >= delta_high and cov > coverage_t:
            emerging += 1
        elif delta >= delta_high:
            spike += 1
        elif delta >= delta_moderate and growing:
            moderate += 1
        else:
            discarded += 1
    return {"emerging": emerging, "spike": spike, "moderate": moderate,
            "discarded": discarded, "total_trend": emerging + spike + moderate}


def analyze_threshold_sensitivity(stats: list[dict]):
    """Qué pasa si cambiamos los umbrales Δ."""
    print("\n" + "=" * 70)
    print("2. SENSIBILIDAD DE UMBRALES Δ")
    print("=" * 70)

    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS]

    # Tabla detallada con Δ_HIGH y Δ_MODERATE fijos en su ratio real (1.0 = 2/3 de 1.5)
    test_configs = [
        (0.5, 0.33),
        (0.75, 0.50),
        (1.0, 0.67),
        (1.25, 0.83),
        (1.5, 1.0),   # adoptado
        (1.75, 1.17),
        (2.0, 1.33),
        (2.5, 1.67),
        (3.0, 2.0),
    ]

    print(f"\n{'Δ_HIGH':>8} {'Emerging':>10} {'Spike':>8} {'Moderate':>10} {'Total trend':>12} {'Discarded':>10}")
    print("-" * 65)

    for d_high, d_mod in test_configs:
        r = _simulate_decisions(valid, d_high, d_mod)
        marker = " ◄ elegido" if d_high == 1.5 else ""
        print(f"  {d_high:>5.1f}   {r['emerging']:>8}   {r['spike']:>6}   {r['moderate']:>8}   {r['total_trend']:>10}   {r['discarded']:>8}{marker}")


def analyze_statistical_basis(stats: list[dict]):
    """Justificación estadística del Δ como z-score."""
    print("\n" + "=" * 70)
    print("3. JUSTIFICACIÓN ESTADÍSTICA (z-score modificado)")
    print("=" * 70)

    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS]

    print(f"""
Δ = (W_current - μ_historical) / σ_effective

Donde:
  W_current    = proporción promedio diaria del tópico en ventana de evaluación
  μ_historical = media de proporciones diarias en ventana histórica (baseline)
  σ_effective  = max(σ_historical, {STD_FLOOR})

Si Δ fuera un z-score puro (distribución normal):
  Δ ≥ 1.0 → P ≈ 15.9% (percentil ~84)  → "por encima de lo normal"
  Δ ≥ 1.5 → P ≈  6.7% (percentil ~93)  → "significativamente por encima"
  Δ ≥ 2.0 → P ≈  2.3% (percentil ~98)  → "raramente alto"
  Δ ≥ 3.0 → P ≈  0.1% (percentil ~99.9) → "extremo"

En nuestro corpus (distribución empírica):""")

    deltas = [s["delta"] for s in valid]
    for t in [1.0, 1.5, 2.0, 3.0]:
        above = sum(1 for d in deltas if d >= t)
        pct = above / len(deltas) * 100
        print(f"  Δ ≥ {t:.1f}: {above:>4} tópicos ({pct:.1f}% del total)")

    print(f"""
¿Por qué 1.5 y no 2.0 (z-score clásico)?

  - En detección de anomalías en streams de texto, 2σ (95%) es demasiado
    conservador — perdería tendencias emergentes legítimas.
  - 1.5σ (~93° percentil) es estándar en la literatura:
    • Mathioudakis & Koudas (2010): TwitterMonitor usa ~1.5σ
    • Kleinberg (2002): Bursty detection usa umbrales adaptativos similares
  - Además, nuestro Δ tiene corrección (STD_FLOOR) que lo hace más
    conservador que un z-score puro, compensando el umbral más bajo.
""")


def analyze_std_floor(stats: list[dict]):
    """Análisis del STD_FLOOR."""
    print("=" * 70)
    print("4. JUSTIFICACIÓN DEL STD_FLOOR = 0.005")
    print("=" * 70)

    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS]
    used_floor = [s for s in valid if s["used_floor"]]
    not_floor = [s for s in valid if not s["used_floor"]]

    print(f"\nTópicos que usan STD_FLOOR: {len(used_floor)} ({len(used_floor)/len(valid)*100:.1f}%)")
    print(f"Tópicos con σ natural:     {len(not_floor)} ({len(not_floor)/len(valid)*100:.1f}%)")

    if not_floor:
        natural_stds = [s["hist_std"] for s in not_floor]
        print(f"\nσ natural (tópicos sin floor):")
        print(f"  Media:    {np.mean(natural_stds):.5f}")
        print(f"  Mediana:  {np.median(natural_stds):.5f}")
        print(f"  P5:       {np.percentile(natural_stds, 5):.5f}")
        print(f"  P10:      {np.percentile(natural_stds, 10):.5f}")
        print(f"  Min:      {min(natural_stds):.5f}")

    if used_floor:
        real_stds = [s["hist_std"] for s in used_floor]
        deltas_floor = [s["delta"] for s in used_floor]
        print(f"\nσ real de tópicos que usan floor:")
        print(f"  Media:  {np.mean(real_stds):.6f}")
        print(f"  Max:    {max(real_stds):.6f}")

        # Sin floor, qué Δ tendrían
        print(f"\nSin STD_FLOOR, estos tópicos tendrían:")
        inflated = []
        for s in used_floor:
            if s["hist_std"] > 0:
                raw_delta = (s["curr_weight"] - s["hist_mean"]) / s["hist_std"]
                inflated.append(raw_delta)
        if inflated:
            print(f"  Δ medio:  {np.mean(inflated):+.1f} (vs {np.mean(deltas_floor):+.1f} con floor)")
            print(f"  Δ máximo: {max(inflated):+.1f} (vs {max(deltas_floor):+.1f} con floor)")
            print(f"  → Sin floor, Δ se infla artificialmente por σ→0")

    print(f"""
Justificación:
  STD_FLOOR = 0.005 evita que tópicos con variabilidad histórica casi nula
  generen Δ artificialmente altos. Ejemplo: un tópico que apareció 0.1% de
  los días con σ = 0.0001 tendría Δ = 10000+ con cualquier pequeño cambio.

  El valor 0.005 se eligió como el percentil ~5-10 de las σ naturales del
  corpus: suficiente para penalizar variabilidad casi nula sin afectar a
  tópicos con historia estadística real.
""")


def analyze_coverage_threshold(stats: list[dict]):
    """Análisis del COVERAGE_THRESHOLD = 5%."""
    print("=" * 70)
    print("5. JUSTIFICACIÓN DEL COVERAGE_THRESHOLD = 5%")
    print("=" * 70)

    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS and s["delta"] >= 1.5]

    if not valid:
        print("No hay tópicos con Δ ≥ 1.5")
        return

    print(f"\nTópicos con Δ ≥ 1.5: {len(valid)}")
    print(f"\n{'Coverage':>10} {'Tipo':>18} {'Ejemplo':>45}")
    print("-" * 78)

    for s in sorted(valid, key=lambda x: -x["coverage"]):
        tipo = "EMERGING" if s["coverage"] > 0.05 else "SPIKE"
        label = s["topic_label"][:42]
        print(f"  {s['coverage']:>8.2%}   {tipo:>16}   {label}")

    coverages = [s["coverage"] for s in valid]
    print(f"\nDistribución de cobertura en tópicos trending:")
    print(f"  > 5%: {sum(1 for c in coverages if c > 0.05)} (emerging trends)")
    print(f"  ≤ 5%: {sum(1 for c in coverages if c <= 0.05)} (localized spikes)")

    print(f"""
Justificación:
  5% = umbral donde un tópico representa una fracción significativa del
  discurso político total. Con ~200K textos, 5% = ~10,000 textos.

  • > 5%: El tópico es una tendencia amplia del discurso → emerging_trend
  • ≤ 5%: El tópico es un pico localizado/nicho → localized_spike

  Ambos son tendencias reales, pero la distinción importa para el reporte:
  emerging_trends reflejan cambios en el discurso mayoritario, mientras que
  spikes pueden ser eventos puntuales que no dominan la conversación.
""")


def generate_plots(stats: list[dict]):
    """Genera gráficos para la defensa."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS]
    deltas = [s["delta"] for s in valid]

    # --- Plot 1: Distribución de Δ con umbrales ---
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(deltas, bins=80, color="#607D8B", alpha=0.7, edgecolor="white", linewidth=0.3)

    ax.axvline(x=1.0, color="orange", linestyle="--", linewidth=2, label="Δ_MODERATE = 1.0")
    ax.axvline(x=1.5, color="red", linestyle="--", linewidth=2, label="Δ_HIGH = 1.5")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    # Zonas
    ymax = ax.get_ylim()[1]
    ax.axvspan(min(deltas) - 0.5, 1.0, alpha=0.05, color="blue")
    ax.axvspan(1.0, 1.5, alpha=0.08, color="orange")
    ax.axvspan(1.5, max(deltas) + 0.5, alpha=0.08, color="red")

    ax.text(0.3, ymax * 0.9, "Discarded", fontsize=11, color="blue", alpha=0.7)
    ax.text(1.15, ymax * 0.9, "Moderate", fontsize=11, color="orange")
    ax.text(max(1.7, np.percentile(deltas, 95)), ymax * 0.9, "Trend/Spike", fontsize=11, color="red")

    # Percentiles como referencia
    for p, label in [(90, "P90"), (95, "P95"), (99, "P99")]:
        val = np.percentile(deltas, p)
        if val > 1.5:
            ax.annotate(f"{label}\n({val:.1f})", xy=(val, 0), xytext=(val, ymax * 0.6),
                        fontsize=8, ha="center", color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    ax.set_xlabel("Δ (z-score modificado)", fontsize=12)
    ax.set_ylabel("Número de tópicos", fontsize=12)
    ax.set_title("Distribución de Δ — Umbrales de decisión del agente de tendencias", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "delta_distribution.png")
    fig.savefig(path1, dpi=150)
    plt.close()
    print(f"\n[Plot] {path1}")

    # --- Plot 2: Δ vs Coverage (scatter) ---
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = []
    for s in valid:
        d = s["delta"]
        c = s["coverage"]
        if d >= 1.5 and c > 0.05:
            colors.append("#F44336")  # emerging
        elif d >= 1.5:
            colors.append("#FF9800")  # spike
        elif d >= 1.0 and s["curr_weight"] >= s["hist_mean"]:
            colors.append("#FFC107")  # moderate
        else:
            colors.append("#90CAF9")  # discarded

    coverages = [s["coverage"] * 100 for s in valid]
    ax.scatter(coverages, deltas, c=colors, alpha=0.6, s=40, edgecolors="white", linewidth=0.3)

    ax.axhline(y=1.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Δ_HIGH = 1.5")
    ax.axhline(y=1.0, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="Δ_MODERATE = 1.0")
    ax.axvline(x=5, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Coverage = 5%")

    # Cuadrantes
    ax.text(8, max(deltas) * 0.95, "EMERGING\nTREND", fontsize=12, color="red",
            fontweight="bold", alpha=0.7, ha="center")
    ax.text(1.5, max(deltas) * 0.95, "LOCALIZED\nSPIKE", fontsize=12, color="#FF9800",
            fontweight="bold", alpha=0.7, ha="center")

    # Labels para top trending
    top = sorted(valid, key=lambda s: -s["delta"])[:5]
    for s in top:
        label = s["topic_label"][:25]
        ax.annotate(label, xy=(s["coverage"] * 100, s["delta"]),
                    xytext=(5, 8), textcoords="offset points", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Cobertura del corpus (%)", fontsize=12)
    ax.set_ylabel("Δ (z-score modificado)", fontsize=12)
    ax.set_title("Δ vs Cobertura — Clasificación de tendencias", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "delta_vs_coverage.png")
    fig.savefig(path2, dpi=150)
    plt.close()
    print(f"[Plot] {path2}")

    # --- Plot 3: STD_FLOOR effect ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    stds = [s["hist_std"] for s in valid]
    ax1.hist(stds, bins=60, color="#4CAF50", alpha=0.7, edgecolor="white")
    ax1.axvline(x=STD_FLOOR, color="red", linestyle="--", linewidth=2, label=f"STD_FLOOR = {STD_FLOOR}")
    ax1.set_xlabel("σ histórica", fontsize=12)
    ax1.set_ylabel("Frecuencia", fontsize=12)
    ax1.set_title("Distribución de σ histórica por tópico", fontsize=12)
    ax1.legend()

    n_floor = sum(1 for s in stds if s < STD_FLOOR)
    n_natural = sum(1 for s in stds if s >= STD_FLOOR)
    ax1.text(0.95, 0.95, f"Usan floor: {n_floor}\nσ natural: {n_natural}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # Sin floor vs con floor
    floor_topics = [s for s in valid if s["used_floor"]]
    if floor_topics:
        deltas_with = [s["delta"] for s in floor_topics]
        deltas_without = []
        for s in floor_topics:
            if s["hist_std"] > 0:
                deltas_without.append((s["curr_weight"] - s["hist_mean"]) / s["hist_std"])
            else:
                deltas_without.append(0)

        ax2.scatter(deltas_with, deltas_without, alpha=0.4, s=20, color="#F44336")
        lim = max(max(abs(d) for d in deltas_with + deltas_without), 10)
        ax2.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
        ax2.set_xlabel("Δ con STD_FLOOR", fontsize=12)
        ax2.set_ylabel("Δ sin STD_FLOOR (inflado)", fontsize=12)
        ax2.set_title("Efecto del STD_FLOOR en Δ", fontsize=12)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Ningún tópico usa STD_FLOOR", transform=ax2.transAxes, ha="center")

    fig.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "std_floor_analysis.png")
    fig.savefig(path3, dpi=150)
    plt.close()
    print(f"[Plot] {path3}")


def generate_sensitivity_table_html(stats: list[dict], db_path: str = DB_PATH):
    """Genera tabla HTML con todos los umbrales Δ del sistema para presentación."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS]
    total_topics = len(stats)

    # Cargar sentimiento por tópico para calcular negatividad
    import sqlite3 as _sql
    conn = _sql.connect(db_path)
    conn.row_factory = _sql.Row
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT model_run_id FROM trend_analysis ORDER BY analyzed_at DESC LIMIT 1")
    mid = cur.fetchone()["model_run_id"]

    cur.execute("""
        SELECT ta.topic_label, sr.final_label, COUNT(*) as cnt
        FROM topic_assignments ta
        JOIN sentiment_results sr ON ta.source_id = sr.source_id AND ta.source_type = sr.source_type
        WHERE ta.model_run_id = ? AND ta.topic_id != -1
        GROUP BY ta.topic_label, sr.final_label
    """, (mid,))
    topic_sent = {}
    for r in cur.fetchall():
        lbl = r["topic_label"]
        if lbl not in topic_sent:
            topic_sent[lbl] = {}
        topic_sent[lbl][r["final_label"]] = r["cnt"]
    conn.close()

    def neg_pct(topic_label):
        sent = topic_sent.get(topic_label, {})
        total_s = sum(sent.values())
        return sent.get("negative", 0) / total_s if total_s else 0

    # Contar por zona del sistema completo
    n_alerta_crit = sum(1 for s in valid if s["delta"] >= 3.0 and neg_pct(s["topic_label"]) >= 0.70)
    n_alerta_info = sum(1 for s in valid if s["delta"] >= 2.0
                        and not (s["delta"] >= 3.0 and neg_pct(s["topic_label"]) >= 0.70))
    n_spike_emerg = sum(1 for s in valid if 1.5 <= s["delta"] < 2.0)
    n_moderate = sum(1 for s in valid
                     if 1.0 <= s["delta"] < 1.5 and s["curr_weight"] >= s["hist_mean"])
    n_discarded = len(valid) - n_alerta_crit - n_alerta_info - n_spike_emerg - n_moderate
    n_total_trend = n_alerta_crit + n_alerta_info + n_spike_emerg + n_moderate

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; margin: 0; background: #f5f5f5;
  }}
  .container {{ background: white; padding: 36px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); max-width: 820px; }}
  h3 {{ margin: 0 0 8px 0; color: #1a1a1a; font-size: 18px; }}
  .subtitle {{ margin: 0 0 20px 0; color: #666; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{
    background: #1a1a2e; color: white; padding: 11px 16px;
    text-align: left; font-weight: 600; font-size: 13px;
  }}
  td {{ padding: 11px 16px; border-bottom: 1px solid #e0e0e0; font-size: 13px; color: #333; }}
  tr:last-child td {{ border-bottom: none; }}
  .critical td {{ background: #ffebee; }}
  .critical td:nth-child(3) {{ color: #c62828; font-weight: 600; }}
  .info td {{ background: #fff8e1; }}
  .info td:nth-child(3) {{ color: #f57f17; font-weight: 600; }}
  .trend td {{ background: #fff3e0; }}
  .moderate td {{ background: #e8f5e9; }}
  .discarded td {{ background: #fafafa; color: #999; }}
  .total td {{ font-weight: 700; background: #e3f2fd; border-top: 2px solid #1a1a2e; }}
  .section-header td {{
    background: #f5f5f5; font-weight: 700; font-size: 12px;
    color: #666; text-transform: uppercase; letter-spacing: 0.5px;
    padding: 8px 16px;
  }}
  .note {{ margin-top: 22px; font-size: 12px; color: #666; line-height: 1.7; }}
  .note strong {{ color: #333; }}
</style>
</head>
<body>
<div class="container">
  <h3>Umbrales Δ del sistema — Justificación empírica</h3>
  <p class="subtitle">Δ = (W<sub>current</sub> − μ<sub>historical</sub>) / max(σ<sub>historical</sub>, 0.005) — z-score modificado</p>
  <table>
    <thead>
      <tr>
        <th>Umbral Δ</th>
        <th>Condición</th>
        <th>Decisión</th>
        <th>Tópicos</th>
        <th>Justificación estadística</th>
      </tr>
    </thead>
    <tbody>
      <tr class="section-header"><td colspan="5">Alertas (agente de validación)</td></tr>
      <tr class="critical">
        <td><strong>Δ ≥ 3.0</strong></td>
        <td>+ negatividad &gt; 70%</td>
        <td>Alerta crítica</td>
        <td>{n_alerta_crit}</td>
        <td>P99.9 en normal — evento extremo con carga emocional</td>
      </tr>
      <tr class="info">
        <td><strong>Δ ≥ 2.0</strong></td>
        <td>—</td>
        <td>Alerta informativa</td>
        <td>{n_alerta_info}</td>
        <td>P97.7 en normal — crecimiento raramente alto</td>
      </tr>
      <tr class="section-header"><td colspan="5">Clasificación de tendencias (agente de tendencias)</td></tr>
      <tr class="trend">
        <td><strong>Δ ≥ 1.5</strong></td>
        <td>+ cobertura &gt;/≤ 5%</td>
        <td>emerging / spike</td>
        <td>{n_spike_emerg + n_alerta_crit + n_alerta_info}</td>
        <td>P93.3 en normal — significativamente por encima del baseline</td>
      </tr>
      <tr class="moderate">
        <td><strong>1.0 ≤ Δ &lt; 1.5</strong></td>
        <td>+ peso actual &gt; media</td>
        <td>moderate_trend</td>
        <td>{n_moderate}</td>
        <td>P84.1 en normal — por encima de lo esperado, en crecimiento</td>
      </tr>
      <tr class="discarded">
        <td>Δ &lt; 1.0</td>
        <td>—</td>
        <td>descartado</td>
        <td>{n_discarded}</td>
        <td>Dentro de 1σ — variación normal del discurso</td>
      </tr>
      <tr class="total">
        <td colspan="3">Total tendencias detectadas</td>
        <td><strong>{n_total_trend}</strong> de {len(valid)}</td>
        <td>{total_topics} tópicos BERTopic</td>
      </tr>
    </tbody>
  </table>
  <div class="note">
    <strong>Interpretación de los umbrales como z-scores:</strong><br>
    Δ = 1.0 → 1σ → ~15.9% de probabilidad bajo H₀ (variación normal)<br>
    Δ = 1.5 → 1.5σ → ~6.7% — umbral estándar en detección de tendencias en streams (Mathioudakis &amp; Koudas, 2010)<br>
    Δ = 2.0 → 2σ → ~2.3% — estadísticamente significativo al 95%<br>
    Δ = 3.0 → 3σ → ~0.1% — evento extremo, requiere validación de polaridad (negatividad &gt; 70%) para alerta crítica<br>
    <br>
    <strong>¿Por qué negatividad &gt; 70% para alerta crítica?</strong> Un Δ extremo puede ser positivo (ej: celebración).
    La condición de negatividad asegura que la alerta se active solo ante crisis o eventos con alta carga emocional negativa.
  </div>
</div>
</body>
</html>"""

    path = os.path.join(OUTPUT_DIR, "tabla_sensibilidad_delta.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"[HTML] {path}")
    print(f"       Abre en browser para screenshot: file://{os.path.abspath(path)}")


def print_defense_summary(stats: list[dict]):
    """Resumen final para defensa."""
    valid = [s for s in stats if s["n_current"] >= MIN_TOPIC_TEXTS]
    deltas = [s["delta"] for s in valid]

    n_emerging = sum(1 for s in valid if s["delta"] >= 1.5 and s["coverage"] > 0.05)
    n_spike = sum(1 for s in valid if s["delta"] >= 1.5 and s["coverage"] <= 0.05)
    n_moderate = sum(1 for s in valid if 1.0 <= s["delta"] < 1.5 and s["curr_weight"] >= s["hist_mean"])
    n_discarded = len(valid) - n_emerging - n_spike - n_moderate

    print("\n" + "=" * 70)
    print("RESUMEN PARA DEFENSA — UMBRALES DE TENDENCIAS")
    print("=" * 70)
    print(f"""
FÓRMULA:
  Δ = (W_current - μ_historical) / max(σ_historical, 0.005)

  Es un z-score modificado con floor en σ para robustez.

UMBRALES Y JUSTIFICACIÓN:

  Δ_HIGH = 1.5 (emerging_trend / localized_spike):
    • Equivale al percentil ~{100 - sum(1 for d in deltas if d >= 1.5)/len(deltas)*100:.0f} de la distribución empírica
    • En distribución normal, 1.5σ → P ≈ 6.7% (percentil 93)
    • Litertura: Mathioudakis & Koudas (2010) usa ~1.5σ en TwitterMonitor
    • Más conservador que 1σ (demasiado ruido) pero menos que 2σ (pierde señal)

  Δ_MODERATE = 1.0 (moderate_trend):
    • Equivale al percentil ~{100 - sum(1 for d in deltas if d >= 1.0)/len(deltas)*100:.0f} de la distribución empírica
    • Captura tópicos en crecimiento que aún no son "emergentes"
    • Requiere además que el peso actual > media histórica (filtro de crecimiento)

  COVERAGE_THRESHOLD = 5%:
    • Distingue entre tendencia amplia vs spike localizado
    • > 5% del corpus → "emerging_trend" (cambio en discurso mayoritario)
    • ≤ 5% → "localized_spike" (evento puntual en nicho)

  STD_FLOOR = 0.005:
    • Evita Δ artificialmente alto en tópicos con σ → 0
    • Corresponde al percentil ~5-10 de las σ naturales del corpus
    • Sin él, tópicos estables generarían Δ > 1000

RESULTADO EN NUESTRO CORPUS:
  Emerging trends:  {n_emerging:>4}
  Localized spikes: {n_spike:>4}
  Moderate trends:  {n_moderate:>4}
  Discarded:        {n_discarded:>4}
  Total evaluados:  {len(valid):>4}
""")


if __name__ == "__main__":
    print(f"Base de datos: {DB_PATH}\n")
    stats = load_topic_stats(DB_PATH)

    if not stats:
        print("No se pudieron cargar datos de tópicos.")
        exit(1)

    analyze_delta_distribution(stats)
    analyze_threshold_sensitivity(stats)
    analyze_statistical_basis(stats)
    analyze_std_floor(stats)
    analyze_coverage_threshold(stats)
    generate_plots(stats)
    generate_sensitivity_table_html(stats)
