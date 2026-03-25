"""
Evaluación de RoBERTa contra pseudo ground truth de DeepSeek.

Pipeline:
  Step 1 — (ya hecho) label_ground_truth.py etiqueta con DeepSeek
  Step 2 — Merge DeepSeek + RoBERTa predictions
  Step 3 — Agreement analysis
  Step 4 — Métricas: accuracy, precision, recall, F1, confusion matrix
  Step 5 — Dataset splits: accepted / disagreement / ambiguous
  Step 6 — Export: labeled_dataset.csv, disagreement_cases.csv, evaluation_metrics.json

Uso:
    python -m scripts.evaluate_ground_truth
    python -m scripts.evaluate_ground_truth --out-dir data/evaluation
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import DB_PATH

AMBIGUOUS_CONF_THRESHOLD = 0.50  # igual que LOW_CONF_THRESHOLD en sentiment_agent.py


def load_data(db_path: str) -> pd.DataFrame:
    """
    Step 2 — Merge ground truth (DeepSeek) con predicciones de RoBERTa.
    Retorna un DataFrame con una fila por texto etiquetado en ambas tablas.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            gt.source_id        AS comment_id,
            gt.source_type,
            gt.original_text    AS text,
            gt.llm_label        AS deepseek_label,
            gt.llm_reasoning    AS reasoning,
            sr.final_label      AS roberta_label,
            sr.roberta_confidence,
            sr.decision         AS roberta_decision
        FROM ground_truth_labels gt
        JOIN sentiment_results sr
            ON gt.source_id = sr.source_id
            AND gt.source_type = sr.source_type
        ORDER BY gt.labeled_at
    """, conn)
    conn.close()
    return df


def add_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 — Agrega columna de acuerdo entre DeepSeek y RoBERTa."""
    df["agreement"] = df["deepseek_label"] == df["roberta_label"]
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Step 4 — Calcula métricas de evaluación.
    Excluye casos donde RoBERTa marcó 'ambiguous' (no hay predicción de clase).
    """
    # Para métricas: solo casos donde RoBERTa hizo una predicción de clase
    eval_df = df[df["roberta_label"] != "ambiguous"].copy()

    y_true = eval_df["deepseek_label"]
    y_pred = eval_df["roberta_label"]
    labels = ["positive", "negative", "neutral"]

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro  = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels,
                                   zero_division=0, output_dict=True)

    metrics = {
        "n_total":         len(df),
        "n_evaluated":     len(eval_df),
        "n_ambiguous_roberta": int((df["roberta_label"] == "ambiguous").sum()),
        "accuracy":        round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro":    round(recall, 4),
        "f1_macro":        round(f1_macro, 4),
        "agreement_rate":  round(df["agreement"].mean(), 4),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall":    round(report[label]["recall"], 4),
                "f1":        round(report[label]["f1-score"], 4),
                "support":   int(report[label]["support"]),
            }
            for label in labels if label in report
        },
    }
    return metrics


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 5 — Crea tres subsets del dataset.

    accepted_dataset    : DeepSeek == RoBERTa (acuerdo total)
    disagreement_dataset: DeepSeek != RoBERTa (desacuerdo)
    ambiguous_dataset   : RoBERTa confidence < threshold O RoBERTa marcó 'ambiguous'
    """
    ambiguous_mask = (
        (df["roberta_confidence"] < AMBIGUOUS_CONF_THRESHOLD) |
        (df["roberta_label"] == "ambiguous")
    )
    accepted_mask     = df["agreement"] & ~ambiguous_mask
    disagreement_mask = ~df["agreement"] & ~ambiguous_mask

    accepted     = df[accepted_mask].copy()
    disagreement = df[disagreement_mask].copy()
    ambiguous    = df[ambiguous_mask].copy()

    return accepted, disagreement, ambiguous


def print_summary(metrics: dict,
                  accepted: pd.DataFrame,
                  disagreement: pd.DataFrame,
                  ambiguous: pd.DataFrame):
    labels_order = ["positive", "negative", "neutral"]
    icons = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}

    print(f"\n{'='*65}")
    print("  EVALUACIÓN: RoBERTa vs DeepSeek (pseudo ground truth)")
    print(f"{'='*65}\n")

    print(f"  Total etiquetados   : {metrics['n_total']:,}")
    print(f"  Evaluables          : {metrics['n_evaluated']:,}")
    print(f"  Ambiguous RoBERTa   : {metrics['n_ambiguous_roberta']:,}")
    print()

    print(f"  Accuracy            : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro)   : {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)      : {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro)          : {metrics['f1_macro']:.4f}")
    print(f"  Agreement rate      : {metrics['agreement_rate']:.1%}")
    print()

    print("  Métricas por clase:")
    for label in labels_order:
        if label in metrics["per_class"]:
            m = metrics["per_class"][label]
            print(f"  {icons[label]} {label:<10}  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}  n={m['support']}")
    print()

    print("  Confusion matrix (rows=true, cols=pred):")
    col_labels = metrics["confusion_matrix"]["labels"]
    print(f"  {'':>12} " + "  ".join(f"{l:>10}" for l in col_labels))
    for i, row_label in enumerate(col_labels):
        row = metrics["confusion_matrix"]["matrix"][i]
        print(f"  {row_label:>12} " + "  ".join(f"{v:>10}" for v in row))
    print()

    print(f"  Dataset splits:")
    print(f"  ✅ accepted_dataset     : {len(accepted):,} textos")
    print(f"  ❌ disagreement_dataset : {len(disagreement):,} textos")
    print(f"  ⚪ ambiguous_dataset    : {len(ambiguous):,} textos")
    print()


def export(df: pd.DataFrame,
           accepted: pd.DataFrame,
           disagreement: pd.DataFrame,
           ambiguous: pd.DataFrame,
           metrics: dict,
           out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # labeled_dataset.csv — dataset completo con todas las columnas
    df.to_csv(out_dir / "labeled_dataset.csv", index=False)
    logger.info(f"  Exportado: {out_dir}/labeled_dataset.csv ({len(df):,} filas)")

    # disagreement_cases.csv — casos de desacuerdo para análisis de errores
    disagreement.to_csv(out_dir / "disagreement_cases.csv", index=False)
    logger.info(f"  Exportado: {out_dir}/disagreement_cases.csv ({len(disagreement):,} filas)")

    # accepted_dataset.csv
    accepted.to_csv(out_dir / "accepted_dataset.csv", index=False)
    logger.info(f"  Exportado: {out_dir}/accepted_dataset.csv ({len(accepted):,} filas)")

    # ambiguous_dataset.csv
    ambiguous.to_csv(out_dir / "ambiguous_dataset.csv", index=False)
    logger.info(f"  Exportado: {out_dir}/ambiguous_dataset.csv ({len(ambiguous):,} filas)")

    # evaluation_metrics.json
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Exportado: {out_dir}/evaluation_metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluación RoBERTa vs DeepSeek ground truth")
    parser.add_argument("--out-dir", type=str, default="data/evaluation",
                        help="Directorio de salida para los CSVs y JSON (default: data/evaluation)")
    args = parser.parse_args()

    # Step 2 — Cargar y mergear
    logger.info("Cargando datos...")
    df = load_data(DB_PATH)

    if df.empty:
        logger.error("No hay datos. Corre primero: python -m scripts.label_ground_truth --all --save")
        return

    logger.info(f"  {len(df):,} textos con ambas etiquetas")

    # Step 3 — Agreement
    df = add_agreement(df)

    # Step 4 — Métricas
    metrics = compute_metrics(df)

    # Step 5 — Splits
    accepted, disagreement, ambiguous = create_splits(df)

    # Imprimir resumen
    print_summary(metrics, accepted, disagreement, ambiguous)

    # Step 6 — Export
    out_dir = Path(args.out_dir)
    logger.info(f"Exportando resultados a {out_dir}/")
    export(df, accepted, disagreement, ambiguous, metrics, out_dir)

    print(f"{'='*65}")
    print(f"  Archivos guardados en: {out_dir}/")
    print(f"    labeled_dataset.csv")
    print(f"    disagreement_cases.csv")
    print(f"    accepted_dataset.csv")
    print(f"    ambiguous_dataset.csv")
    print(f"    evaluation_metrics.json")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
