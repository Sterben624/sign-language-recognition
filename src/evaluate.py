import time
from dataclasses import dataclass, field
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.8

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from tensorflow.keras import Model


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    inference_ms_per_sample: float
    model_size_mb: float
    per_class_report: dict
    confusion_matrix: np.ndarray
    train_time_sec: float = 0.0
    covered_accuracy: float = 0.0  # accuracy по предсказаниям >= CONFIDENCE_THRESHOLD
    coverage: float = 1.0          # доля семплов выше порога


class ImageModelEvaluator:
    def __init__(self, model: Model, class_names: list[str], model_path: Path):
        self.model = model
        self.class_names = class_names
        self.model_path = model_path

    def evaluate(self, test_gen, model_name: str, train_time_sec: float = 0.0) -> EvaluationResult:
        test_gen.reset()
        n_samples = len(test_gen.filenames)

        start = time.perf_counter()
        probs = self.model.predict(test_gen, verbose=0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        y_pred = np.argmax(probs, axis=1)
        y_true = test_gen.classes
        inference_ms = elapsed_ms / n_samples

        return _build_result(model_name, y_true, y_pred, probs, inference_ms,
                             self.model_path, self.class_names, train_time_sec)


class LandmarkModelEvaluator:
    def __init__(self, model: Model, class_names: list[str], model_path: Path):
        self.model = model
        self.class_names = class_names
        self.model_path = model_path

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 model_name: str, train_time_sec: float = 0.0) -> EvaluationResult:
        start = time.perf_counter()
        probs = self.model.predict(X_test, verbose=0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        y_pred = np.argmax(probs, axis=1)
        inference_ms = elapsed_ms / len(y_test)
        return _build_result(model_name, y_test, y_pred, probs, inference_ms,
                             self.model_path, self.class_names, train_time_sec)


class RandomForestEvaluator:
    def __init__(self, model_path: Path, class_names: list[str]):
        self.model = joblib.load(model_path)
        self.class_names = class_names
        self.model_path = model_path

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 train_time_sec: float = 0.0) -> EvaluationResult:
        start = time.perf_counter()
        probs = self.model.predict_proba(X_test)
        elapsed_ms = (time.perf_counter() - start) * 1000

        y_pred = np.argmax(probs, axis=1)
        inference_ms = elapsed_ms / len(y_test)
        return _build_result("random_forest", y_test, y_pred, probs, inference_ms,
                             self.model_path, self.class_names, train_time_sec)


class ComparisonReport:
    def __init__(self, results: list[EvaluationResult], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics_table(self) -> pd.DataFrame:
        rows = [
            {
                "model": r.model_name,
                "accuracy": round(r.accuracy, 4),
                "macro_precision": round(r.macro_precision, 4),
                "macro_recall": round(r.macro_recall, 4),
                "macro_f1": round(r.macro_f1, 4),
                "inference_ms": round(r.inference_ms_per_sample, 3),
                "model_size_mb": round(r.model_size_mb, 2),
                "train_time_sec": round(r.train_time_sec, 1),
            }
            for r in self.results
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "metrics_comparison.csv", index=False)
        return df

    def plot_confusion_matrices(self) -> None:
        for result in self.results:
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(
                result.confusion_matrix,
                annot=True, fmt="d", cmap="Blues",
                xticklabels=result.per_class_report["classes"],
                yticklabels=result.per_class_report["classes"],
                ax=ax,
            )
            ax.set_title(f"Confusion Matrix — {result.model_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            fig.savefig(self.output_dir / f"confusion_matrix_{result.model_name}.png", dpi=150)
            plt.close(fig)

    def plot_metrics_comparison(self) -> None:
        metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
        names = [r.model_name for r in self.results]
        values = {m: [getattr(r, m) for r in self.results] for m in metrics}

        fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
        for ax, metric in zip(axes, metrics):
            bars = ax.bar(names, values[metric], color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylim(0, 1.05)
            ax.bar_label(bars, fmt="%.3f", padding=3)
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        fig.savefig(self.output_dir / "metrics_comparison.png", dpi=150)
        plt.close(fig)

    def plot_training_history(self, history_csv_paths: dict[str, Path]) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, path in history_csv_paths.items():
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "accuracy" in df.columns:
                axes[0].plot(df["accuracy"], label=f"{name} train")
            if "val_accuracy" in df.columns:
                axes[0].plot(df["val_accuracy"], label=f"{name} val", linestyle="--")
            if "loss" in df.columns:
                axes[1].plot(df["loss"], label=f"{name} train")
            if "val_loss" in df.columns:
                axes[1].plot(df["val_loss"], label=f"{name} val", linestyle="--")

        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(self.output_dir / "training_history.png", dpi=150)
        plt.close(fig)


def _build_result(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    inference_ms: float,
    model_path: Path,
    class_names: list[str],
    train_time_sec: float,
) -> EvaluationResult:
    report = classification_report(y_true, y_pred, target_names=class_names, labels=list(range(len(class_names))), output_dict=True)

    mask = probs.max(axis=1) >= CONFIDENCE_THRESHOLD
    coverage = float(mask.mean())
    covered_accuracy = float(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else 0.0

    return EvaluationResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        macro_precision=report["macro avg"]["precision"],
        macro_recall=report["macro avg"]["recall"],
        macro_f1=report["macro avg"]["f1-score"],
        inference_ms_per_sample=inference_ms,
        model_size_mb=_model_size_mb(model_path),
        per_class_report={"classes": class_names, **report},
        confusion_matrix=confusion_matrix(y_true, y_pred),
        train_time_sec=train_time_sec,
        covered_accuracy=covered_accuracy,
        coverage=coverage,
    )


def _model_size_mb(path: Path) -> float:
    if path.exists():
        return path.stat().st_size / (1024 ** 2)
    return 0.0
