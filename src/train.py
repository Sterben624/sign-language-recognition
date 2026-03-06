import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger,
)

from data_loader import ASLDataLoader, DataConfig
from models.mediapipe_classifier import LandmarkExtractor, MediaPipeConfig, build_landmark_classifier, build_random_forest
from models.mobilenet import MobileNetConfig, build_mobilenet, unfreeze_for_finetuning
from models.simple_cnn import CNNConfig, build_simple_cnn


@dataclass
class TrainConfig:
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
    epochs: int = 25
    finetune_epochs: int = 15
    patience: int = 5
    min_delta: float = 1e-4


class ImageModelTrainer:
    def __init__(self, model: Model, name: str, config: TrainConfig):
        self.model = model
        self.name = name
        self.config = config
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.results_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_gen, val_gen, epochs: int | None = None) -> pd.DataFrame:
        epochs = epochs or self.config.epochs
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=self._callbacks(),
        )
        df = pd.DataFrame(history.history)
        df.to_csv(self.config.results_dir / f"{self.name}_history.csv", index=False)
        return df

    def _callbacks(self) -> list[Callback]:
        return [
            EarlyStopping(
                monitor="val_accuracy",
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=str(self.config.models_dir / f"{self.name}_best.h5"),
                monitor="val_accuracy",
                save_best_only=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
            ),
            CSVLogger(str(self.config.results_dir / f"{self.name}_log.csv")),
        ]


class MobileNetTrainer(ImageModelTrainer):
    def __init__(self, mobilenet_config: MobileNetConfig, train_config: TrainConfig):
        model = build_mobilenet(mobilenet_config)
        super().__init__(model, "mobilenet", train_config)
        self.mobilenet_config = mobilenet_config

    def train_with_finetuning(self, train_gen, val_gen) -> tuple[pd.DataFrame, pd.DataFrame]:
        head_history = self.train(train_gen, val_gen, epochs=self.config.epochs)

        unfreeze_for_finetuning(self.model, self.mobilenet_config)
        finetune_history = self.train(train_gen, val_gen, epochs=self.config.finetune_epochs)

        return head_history, finetune_history


class LandmarkModelTrainer:
    def __init__(self, config: TrainConfig, mediapipe_config: MediaPipeConfig):
        self.config = config
        self.model = build_landmark_classifier(mediapipe_config)
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.results_dir.mkdir(parents=True, exist_ok=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=64,
            callbacks=self._callbacks(),
        )
        df = pd.DataFrame(history.history)
        df.to_csv(self.config.results_dir / "mediapipe_dense_history.csv", index=False)
        return df

    def save(self) -> None:
        self.model.save(str(self.config.models_dir / "mediapipe_dense_best.h5"))

    def _callbacks(self) -> list[Callback]:
        return [
            EarlyStopping(monitor="val_accuracy", patience=self.config.patience,
                          restore_best_weights=True),
            ModelCheckpoint(str(self.config.models_dir / "mediapipe_dense_best.h5"),
                            monitor="val_accuracy", save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ]


class RandomForestTrainer:
    def __init__(self, config: TrainConfig, mediapipe_config: MediaPipeConfig):
        self.config = config
        self.model = build_random_forest(mediapipe_config)
        config.models_dir.mkdir(parents=True, exist_ok=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        self.model.fit(X_train, y_train)
        return self.model

    def save(self) -> None:
        joblib.dump(self.model, self.config.models_dir / "random_forest.pkl")


def extract_and_cache_landmarks(
    loader: ASLDataLoader,
    extractor: LandmarkExtractor,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    cache_dir: Path,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = []

    for name, df in zip(("train", "val", "test"), splits):
        X_path = cache_dir / f"X_{name}.npy"
        y_path = cache_dir / f"y_{name}.npy"

        if X_path.exists() and y_path.exists():
            X, y = np.load(X_path), np.load(y_path)
        else:
            X, y, _ = extractor.process_dataframe(df)
            np.save(X_path, X)
            np.save(y_path, y)

        result.append((X, y))

    return tuple(result)
