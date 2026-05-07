"""
Componente de reporte del Agente de Validación.

Produce gráficos con matplotlib y reportes en Markdown.
Parte integral del agente de validación y reporte (ReAct).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from loguru import logger

from config.settings import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Paleta de colores consistente
SENTIMENT_COLORS = {
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "positive": "#2ecc71",
    "ambiguous": "#f39c12",
}

TREND_COLORS = {
    "emerging_trend": "#e74c3c",
    "moderate_trend": "#f39c12",
    "localized_spike": "#3498db",
    "discarded": "#bdc3c7",
}


class ReportGenerator:
    """Genera gráficos y reportes Markdown."""

    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = REPORTS_DIR / f"report_{ts}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures: list[str] = []
        logger.info(f"ReportGenerator → {self.output_dir}")

    # ------------------------------------------------------------------
    # Gráficos de sentimiento
    # ------------------------------------------------------------------

    def plot_sentiment_distribution(
        self, data: dict[str, int], title: str = "Distribucion de Sentimiento"
    ) -> Path:
        """Gráfico de barras horizontales con distribución de sentimiento."""
        labels = list(data.keys())
        counts = list(data.values())
        total = sum(counts)
        colors = [SENTIMENT_COLORS.get(l, "#7f8c8d") for l in labels]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels, counts, color=colors, edgecolor="white", height=0.6)

        for bar, count in zip(bars, counts):
            pct = count / total * 100 if total else 0
            ax.text(
                bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=10,
            )

        ax.set_xlabel("Textos")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.invert_yaxis()
        plt.tight_layout()

        path = self.output_dir / "sentiment_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path.name)
        return path

    def plot_sentiment_by_topic(
        self, data: dict[str, dict[str, int]],
        title: str = "Sentimiento por Topico Trending",
    ) -> Path:
        """Barras apiladas: sentimiento por tópico.

        Args:
            data: {topic_label: {positive: n, negative: n, neutral: n, ...}}
        """
        topics = list(data.keys())
        sentiment_labels = ["negative", "neutral", "positive", "ambiguous"]
        present = [s for s in sentiment_labels if any(data[t].get(s, 0) > 0 for t in topics)]

        fig, ax = plt.subplots(figsize=(10, max(4, len(topics) * 0.5 + 1)))
        y = np.arange(len(topics))
        left = np.zeros(len(topics))

        for sent in present:
            vals = [data[t].get(sent, 0) for t in topics]
            ax.barh(y, vals, left=left, label=sent.capitalize(),
                    color=SENTIMENT_COLORS.get(sent, "#7f8c8d"), height=0.6)
            left += vals

        ax.set_yticks(y)
        ax.set_yticklabels([t[:40] for t in topics], fontsize=9)
        ax.set_xlabel("Textos")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()

        path = self.output_dir / "sentiment_by_topic.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path.name)
        return path

    def plot_confidence_distribution(
        self, confidences: list[float],
        title: str = "Distribucion de Confianza (RoBERTa)",
    ) -> Path:
        """Histograma de confianza del modelo."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(confidences, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(0.85, color="#e74c3c", linestyle="--", label="High (0.85)")
        ax.axvline(0.50, color="#f39c12", linestyle="--", label="Low (0.50)")
        ax.set_xlabel("Confianza")
        ax.set_ylabel("Frecuencia")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()

        path = self.output_dir / "confidence_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path.name)
        return path

    # ------------------------------------------------------------------
    # Gráficos de tendencias
    # ------------------------------------------------------------------

    def plot_trend_deltas(
        self, topics: list[dict],
        title: str = "Delta por Topico (Tendencias)",
    ) -> Path:
        """Barras horizontales con Δ por tópico, coloreadas por decisión.

        Args:
            topics: lista de dicts con keys: topic_label, delta, trend_decision
        """
        topics = sorted(topics, key=lambda t: t["delta"], reverse=True)[:20]
        labels = [t["topic_label"][:40] for t in topics]
        deltas = [t["delta"] for t in topics]
        colors = [TREND_COLORS.get(t["trend_decision"], "#7f8c8d") for t in topics]

        fig, ax = plt.subplots(figsize=(10, max(4, len(topics) * 0.4 + 1)))
        bars = ax.barh(labels, deltas, color=colors, edgecolor="white", height=0.6)
        ax.axvline(1.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Δ=1.5 (emerging)")
        ax.axvline(1.0, color="#f39c12", linestyle="--", alpha=0.5, label="Δ=1.0 (moderate)")
        ax.set_xlabel("Δ (desviaciones estándar)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.invert_yaxis()
        plt.tight_layout()

        path = self.output_dir / "trend_deltas.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path.name)
        return path

    def plot_daily_trend(
        self, daily_weights: dict[str, float], topic_label: str,
    ) -> Path:
        """Línea temporal de peso diario de un tópico."""
        import matplotlib.dates as mdates
        from datetime import datetime as dt

        dates = sorted(daily_weights.keys())
        weights = [daily_weights[d] for d in dates]

        # Convertir strings a datetime para mejor control del eje X
        date_objs = [dt.strptime(d, "%Y-%m-%d") if isinstance(d, str) else d for d in dates]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(date_objs, weights, marker="o", color="#3498db", linewidth=2, markersize=3)
        ax.fill_between(date_objs, weights, alpha=0.15, color="#3498db")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Peso (proporcion)")
        ax.set_title(f"Evolucion Temporal: {topic_label[:50]}", fontsize=12, fontweight="bold")

        # Mostrar fechas cada 2 semanas para evitar amontonamiento
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()

        safe_name = topic_label[:30].replace(" ", "_").replace("/", "_")
        path = self.output_dir / f"trend_daily_{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path.name)
        return path

    # ------------------------------------------------------------------
    # Word clouds
    # ------------------------------------------------------------------

    def plot_wordcloud(
        self, texts: list[str], topic_label: str,
    ) -> Path:
        """Genera un word cloud a partir de textos de un tópico."""
        from wordcloud import WordCloud

        combined = " ".join(texts)
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="RdYlBu_r",
            max_words=100,
            collocations=False,
            stopwords=None,  # ya preprocesado
        ).generate(combined)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud: {topic_label[:50]}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        safe_name = topic_label[:30].replace(" ", "_").replace("/", "_")
        path = self.output_dir / f"wordcloud_{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path.name)
        return path

    # ------------------------------------------------------------------
    # Tabla comparativa
    # ------------------------------------------------------------------

    def plot_comparison_table(
        self, rows: list[dict], title: str = "Comparacion de Enfoques",
    ) -> Path:
        """Tabla visual comparando métricas de distintos enfoques.

        Args:
            rows: lista de dicts con keys: approach, accuracy, f1_macro, precision_macro, recall_macro
        """
        col_labels = ["Enfoque", "Accuracy", "F1 Macro", "Precision", "Recall"]
        cell_data = []
        for r in rows:
            cell_data.append([
                r["approach"],
                f"{r.get('accuracy', 0):.4f}",
                f"{r.get('f1_macro', 0):.4f}",
                f"{r.get('precision_macro', 0):.4f}",
                f"{r.get('recall_macro', 0):.4f}",
            ])

        fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.5))
        ax.axis("off")
        table = ax.table(
            cellText=cell_data, colLabels=col_labels,
            cellLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)

        # Header color
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor("#2c3e50")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        # Highlight best accuracy row
        if rows:
            best_idx = max(range(len(rows)), key=lambda i: rows[i].get("accuracy", 0))
            for j in range(len(col_labels)):
                table[(best_idx + 1, j)].set_facecolor("#d5f5e3")

        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()

        path = self.output_dir / "comparison_table.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(path.name)
        return path

    # ------------------------------------------------------------------
    # Generador de Markdown
    # ------------------------------------------------------------------

    def generate_markdown_report(
        self, sections: list[dict], title: str = "Reporte de Analisis",
    ) -> Path:
        """Genera un reporte .md combinando texto y referencias a imágenes.

        Args:
            sections: lista de dicts con keys:
                - heading (str)
                - text (str, optional): párrafo de texto
                - image (str, optional): nombre de archivo de imagen
                - table (list[list[str]], optional): filas de tabla markdown
        """
        lines = [f"# {title}", ""]
        lines.append(f"*Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")

        for section in sections:
            lines.append(f"## {section['heading']}")
            lines.append("")

            if "text" in section and section["text"]:
                lines.append(section["text"])
                lines.append("")

            if "image" in section and section["image"]:
                lines.append(f"![{section['heading']}]({section['image']})")
                lines.append("")

            if "table" in section and section["table"]:
                header = section["table"][0]
                lines.append("| " + " | ".join(header) + " |")
                lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                for row in section["table"][1:]:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")
                lines.append("")

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        # Guardar metadata como JSON
        meta = {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "figures": self.figures,
            "output_dir": str(self.output_dir),
        }
        meta_path = self.output_dir / "summary.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Reporte generado: {report_path}")
        return report_path
