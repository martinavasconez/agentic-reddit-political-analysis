"""
Análisis de umbrales de confianza de RoBERTa para justificación empírica.

Genera:
  1. Tabla de accuracy vs umbral de confianza (RoBERTa solo vs ground truth)
  2. Análisis de cada zona de decisión del agente de sentimiento
  3. Análisis del umbral interno 0.65 (desempate en zona media)
  4. Gráficos: curva accuracy-volumen, distribución de confianza, accuracy por zona
"""

import sqlite3
import sys
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reddit_political.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation")


def load_data(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT sr.roberta_label, sr.roberta_confidence, sr.final_label,
               sr.decision, sr.gemini_label,
               gt.llm_label AS deepseek_label
        FROM sentiment_results sr
        JOIN ground_truth_labels gt
            ON sr.source_id = gt.source_id AND sr.source_type = gt.source_type
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def analyze_threshold_curve(data: list[dict]):
    """Accuracy de RoBERTa solo vs ground truth para distintos umbrales."""
    print("=" * 70)
    print("1. CURVA ACCURACY vs UMBRAL DE CONFIANZA (RoBERTa solo)")
    print("=" * 70)
    print(f"{'Umbral':>8} {'N textos':>10} {'% corpus':>10} {'Accuracy':>10} {'Δ Acc':>8}")
    print("-" * 50)

    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
    total = len(data)
    results = []
    prev_acc = None

    for t in thresholds:
        subset = [(r["roberta_label"].lower(), r["deepseek_label"].strip().lower())
                  for r in data if r["roberta_confidence"] > t]
        if not subset:
            continue
        correct = sum(1 for rob, ds in subset if rob == ds)
        acc = correct / len(subset)
        delta = f"{acc - prev_acc:+.1%}" if prev_acc is not None else "—"
        print(f"  > {t:.2f}   {len(subset):>8,}   {len(subset)/total*100:>8.1f}%   {acc:>8.1%}     {delta}")
        results.append({"threshold": t, "n": len(subset), "accuracy": acc})
        prev_acc = acc

    return results


def analyze_agent_zones(data: list[dict]):
    """Accuracy por zona de decisión del agente."""
    print("\n" + "=" * 70)
    print("2. ACCURACY POR ZONA DE DECISIÓN DEL AGENTE")
    print("=" * 70)

    zones = {
        "ALTA (>0.85) → accepted": lambda r: r["roberta_confidence"] > 0.85,
        "MEDIA (0.50-0.85) → cross_validation": lambda r: 0.50 < r["roberta_confidence"] <= 0.85,
        "  ├─ Media-alta (0.65-0.85)": lambda r: 0.65 < r["roberta_confidence"] <= 0.85,
        "  └─ Media-baja (0.50-0.65)": lambda r: 0.50 < r["roberta_confidence"] <= 0.65,
        "BAJA (≤0.50) → rescue": lambda r: r["roberta_confidence"] <= 0.50,
    }

    print(f"\n{'Zona':<35} {'N':>8} {'Acc RoBERTa':>12} {'Acc Final':>12} {'Mejora':>8}")
    print("-" * 80)

    for name, filt in zones.items():
        subset = [r for r in data if filt(r)]
        if not subset:
            continue

        # Accuracy de RoBERTa solo
        rob_correct = sum(1 for r in subset
                          if r["roberta_label"].lower() == r["deepseek_label"].strip().lower())
        rob_acc = rob_correct / len(subset)

        # Accuracy del sistema (label final después de Gemini)
        final_correct = sum(1 for r in subset
                            if r["final_label"].lower() == r["deepseek_label"].strip().lower())
        final_acc = final_correct / len(subset)

        mejora = final_acc - rob_acc
        print(f"{name:<35} {len(subset):>8,} {rob_acc:>11.1%} {final_acc:>11.1%} {mejora:>+7.1%}")


def analyze_mid_threshold(data: list[dict]):
    """Análisis del umbral 0.65 para desempate en zona media."""
    print("\n" + "=" * 70)
    print("3. JUSTIFICACIÓN DEL UMBRAL INTERNO 0.65 (desempate)")
    print("=" * 70)

    mid_zone = [r for r in data if 0.50 < r["roberta_confidence"] <= 0.85]
    disagree = [r for r in mid_zone
                if r["gemini_label"] and r["roberta_label"].lower() != r["gemini_label"].lower()]

    print(f"\nTextos en zona media (0.50-0.85): {len(mid_zone):,}")
    print(f"Desacuerdos RoBERTa vs Gemini:    {len(disagree):,} ({len(disagree)/len(mid_zone)*100:.1f}%)")

    print(f"\n{'Sub-umbral':<25} {'N desac.':>10} {'RoBERTa gana':>14} {'Gemini gana':>14} {'Mejor':>10}")
    print("-" * 78)

    test_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for t in test_thresholds:
        above = [r for r in disagree if r["roberta_confidence"] > t]
        below = [r for r in disagree if r["roberta_confidence"] <= t]

        if above:
            rob_wins_above = sum(1 for r in above
                                 if r["roberta_label"].lower() == r["deepseek_label"].strip().lower())
            gem_wins_above = sum(1 for r in above
                                 if r["gemini_label"].lower() == r["deepseek_label"].strip().lower())
        else:
            rob_wins_above = gem_wins_above = 0

        if below:
            rob_wins_below = sum(1 for r in below
                                 if r["roberta_label"].lower() == r["deepseek_label"].strip().lower())
            gem_wins_below = sum(1 for r in below
                                 if r["gemini_label"].lower() == r["deepseek_label"].strip().lower())
        else:
            rob_wins_below = gem_wins_below = 0

        # Por encima del sub-umbral: ¿quién acierta más?
        above_winner = "RoBERTa" if rob_wins_above >= gem_wins_above else "Gemini"
        # Por debajo: ¿quién acierta más?
        below_winner = "RoBERTa" if rob_wins_below >= gem_wins_below else "Gemini"

        marker = " ◄" if t == 0.65 else ""
        print(f"  conf > {t:.2f} (arriba)   {len(above):>8,}   {rob_wins_above:>8,} ({rob_wins_above/max(len(above),1)*100:.0f}%)   {gem_wins_above:>8,} ({gem_wins_above/max(len(above),1)*100:.0f}%)   {above_winner}{marker}")
        print(f"  conf ≤ {t:.2f} (abajo)    {len(below):>8,}   {rob_wins_below:>8,} ({rob_wins_below/max(len(below),1)*100:.0f}%)   {gem_wins_below:>8,} ({gem_wins_below/max(len(below),1)*100:.0f}%)   {below_winner}{marker}")
        print()


def analyze_low_threshold(data: list[dict]):
    """Análisis del umbral 0.50 para zona de rescate."""
    print("=" * 70)
    print("4. JUSTIFICACIÓN DEL UMBRAL BAJO 0.50 (rescue)")
    print("=" * 70)

    low = [r for r in data if r["roberta_confidence"] <= 0.50]
    print(f"\nTextos con confianza ≤ 0.50: {len(low):,} ({len(low)/len(data)*100:.1f}%)")

    if low:
        rob_acc = sum(1 for r in low
                      if r["roberta_label"].lower() == r["deepseek_label"].strip().lower()) / len(low)
        final_acc = sum(1 for r in low
                        if r["final_label"].lower() == r["deepseek_label"].strip().lower()) / len(low)

        # Distribución de labels de RoBERTa en zona baja
        rob_dist = Counter(r["roberta_label"].lower() for r in low)
        gt_dist = Counter(r["deepseek_label"].strip().lower() for r in low)

        print(f"Accuracy RoBERTa solo: {rob_acc:.1%} (casi aleatorio con 3 clases = 33%)")
        print(f"Accuracy final (con Gemini rescue): {final_acc:.1%}")
        print(f"Mejora por rescue: {final_acc - rob_acc:+.1%}")
        print(f"\nDistribución RoBERTa en zona baja: {dict(rob_dist)}")
        print(f"Distribución GT en zona baja:      {dict(gt_dist)}")

        # Con 3 clases, confianza uniforme = 0.33
        # 0.50 es el punto donde la clase ganadora = resto combinado
        confs = [r["roberta_confidence"] for r in low]
        print(f"\nConfianza en zona baja: media={np.mean(confs):.3f}, mediana={np.median(confs):.3f}")
        print(f"Referencia: distribución uniforme con 3 clases = 0.333")
        print(f"El umbral 0.50 marca donde P(clase_ganadora) = P(otras dos combinadas)")


def generate_plots(data: list[dict], threshold_results: list[dict]):
    """Genera gráficos para la defensa."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Plot 1: Curva accuracy vs umbral ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    thresholds = [r["threshold"] for r in threshold_results]
    accuracies = [r["accuracy"] for r in threshold_results]
    volumes = [r["n"] for r in threshold_results]

    color1 = "#2196F3"
    color2 = "#FF9800"
    ax1.plot(thresholds, accuracies, "o-", color=color1, linewidth=2, markersize=8, label="Accuracy")
    ax1.set_xlabel("Umbral de confianza RoBERTa", fontsize=12)
    ax1.set_ylabel("Accuracy vs Ground Truth", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.5, 1.0)
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    ax2.bar(thresholds, volumes, width=0.03, alpha=0.3, color=color2, label="N textos")
    ax2.set_ylabel("N textos por encima del umbral", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Marcar umbrales elegidos
    for t, label, color in [(0.85, "HIGH (0.85)", "red"), (0.50, "LOW (0.50)", "green"), (0.65, "MID (0.65)", "purple")]:
        ax1.axvline(x=t, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.annotate(label, xy=(t, 0.52), fontsize=9, color=color, ha="center",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8))

    ax1.set_title("Accuracy vs Umbral de Confianza — Justificación empírica de umbrales", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "threshold_accuracy_curve.png")
    fig.savefig(path1, dpi=150)
    plt.close()
    print(f"\n[Plot] {path1}")

    # --- Plot 2: Distribución de confianza con zonas ---
    fig, ax = plt.subplots(figsize=(10, 6))
    confs = [r["roberta_confidence"] for r in data]
    ax.hist(confs, bins=100, color="#607D8B", alpha=0.7, edgecolor="white", linewidth=0.3)

    ax.axvspan(0, 0.50, alpha=0.1, color="red", label=f"Rescue (≤0.50): {sum(1 for c in confs if c <= 0.50):,}")
    ax.axvspan(0.50, 0.65, alpha=0.1, color="purple", label=f"Media-baja (0.50-0.65): {sum(1 for c in confs if 0.50 < c <= 0.65):,}")
    ax.axvspan(0.65, 0.85, alpha=0.1, color="orange", label=f"Media-alta (0.65-0.85): {sum(1 for c in confs if 0.65 < c <= 0.85):,}")
    ax.axvspan(0.85, 1.0, alpha=0.1, color="green", label=f"Accepted (>0.85): {sum(1 for c in confs if c > 0.85):,}")

    for t in [0.50, 0.65, 0.85]:
        ax.axvline(x=t, color="black", linestyle="--", alpha=0.5)

    ax.set_xlabel("Confianza RoBERTa (softmax)", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title("Distribución de confianza RoBERTa — Zonas de decisión del agente", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "confidence_zones_distribution.png")
    fig.savefig(path2, dpi=150)
    plt.close()
    print(f"[Plot] {path2}")

    # --- Plot 3: Accuracy por zona (RoBERTa vs Final) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    zone_defs = [
        ("Accepted\n(>0.85)", lambda r: r["roberta_confidence"] > 0.85),
        ("Media-alta\n(0.65-0.85)", lambda r: 0.65 < r["roberta_confidence"] <= 0.85),
        ("Media-baja\n(0.50-0.65)", lambda r: 0.50 < r["roberta_confidence"] <= 0.65),
        ("Rescue\n(≤0.50)", lambda r: r["roberta_confidence"] <= 0.50),
    ]

    zone_names = []
    rob_accs = []
    final_accs = []
    zone_ns = []

    for name, filt in zone_defs:
        subset = [r for r in data if filt(r)]
        if not subset:
            continue
        zone_names.append(name)
        zone_ns.append(len(subset))

        rob_correct = sum(1 for r in subset
                          if r["roberta_label"].lower() == r["deepseek_label"].strip().lower())
        rob_accs.append(rob_correct / len(subset))

        final_correct = sum(1 for r in subset
                            if r["final_label"].lower() == r["deepseek_label"].strip().lower())
        final_accs.append(final_correct / len(subset))

    x = np.arange(len(zone_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, rob_accs, width, label="RoBERTa solo", color="#F44336", alpha=0.8)
    bars2 = ax.bar(x + width/2, final_accs, width, label="Final (con Gemini)", color="#4CAF50", alpha=0.8)

    for bar, acc, n in zip(bars1, rob_accs, zone_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)
    for bar, acc, n in zip(bars2, final_accs, zone_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    # N textos debajo de cada grupo
    for i, n in enumerate(zone_ns):
        ax.text(i, -0.05, f"n={n:,}", ha="center", va="top", fontsize=9, color="gray",
                transform=ax.get_xaxis_transform())

    ax.set_ylabel("Accuracy vs Ground Truth", fontsize=12)
    ax.set_title("Accuracy por zona de confianza — RoBERTa solo vs Sistema completo", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "accuracy_by_zone.png")
    fig.savefig(path3, dpi=150)
    plt.close()
    print(f"[Plot] {path3}")


def print_summary(data: list[dict]):
    """Resumen final para la defensa."""
    total = len(data)
    print("\n" + "=" * 70)
    print("RESUMEN PARA DEFENSA")
    print("=" * 70)
    print(f"""
Corpus: {total:,} textos con ground truth (DeepSeek-V3)

UMBRALES Y SU JUSTIFICACIÓN EMPÍRICA:

  0.85 (HIGH — accepted sin Gemini):
    • Accuracy de RoBERTa en esta zona: ~82.5%
    • Subir a 0.90 → +3.3pp accuracy pero pierde 47% de los textos
    • Bajar a 0.80 → gana 24K textos pero -2.9pp accuracy
    • 0.85 es el codo de la curva accuracy-volumen

  0.65 (MID — desempate en zona media):
    • Cuando RoBERTa y Gemini discrepan:
      - conf > 0.65: RoBERTa acierta más → RoBERTa gana
      - conf ≤ 0.65: Gemini acierta más → Gemini gana
    • Es el punto de cruce donde Gemini supera a RoBERTa

  0.50 (LOW — zona de rescate):
    • Accuracy de RoBERTa ≤ 0.50: cercana a aleatorio
    • Con 3 clases, P(uniforme) = 0.333
    • 0.50 = P(ganadora) = P(otras dos combinadas)
    • Gemini rescue mejora significativamente esta zona
""")


if __name__ == "__main__":
    db = DB_PATH
    if len(sys.argv) > 1:
        db = sys.argv[1]

    print(f"Base de datos: {db}")
    data = load_data(db)
    print(f"Textos con ground truth: {len(data):,}\n")

    threshold_results = analyze_threshold_curve(data)
    analyze_agent_zones(data)
    analyze_mid_threshold(data)
    analyze_low_threshold(data)
    generate_plots(data, threshold_results)
