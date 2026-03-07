from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tqdm import tqdm


NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURE_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63


@dataclass
class MediaPipeConfig:
    num_classes: int = 29
    dropout_rate: float = 0.3
    l2_lambda: float = 1e-4
    learning_rate: float = 1e-3
    min_detection_confidence: float = 0.5
    n_estimators: int = 200  # для RandomForest
    static_image_mode: bool = True


_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task"
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        import urllib.request
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Завантаження hand_landmarker.task...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Завантажено.")
    return _MODEL_PATH


class LandmarkExtractor:
    def __init__(self, config: MediaPipeConfig):
        self.config = config
        model_path = _ensure_model()
        base_options = _mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = _mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=config.min_detection_confidence,
        )
        self._landmarker = _mp_vision.HandLandmarker.create_from_options(options)

    def extract_from_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return self._normalize(features).flatten()

    def extract_from_frame(self, image_bgr: np.ndarray) -> np.ndarray | None:
        return self.extract_from_image(image_bgr)

    def process_dataframe(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
        features, labels, valid_idx = [], [], []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting landmarks"):
            img = cv2.imread(row["filepath"])
            if img is None:
                continue
            feat = self.extract_from_image(img)
            if feat is None:
                continue
            features.append(feat)
            labels.append(row["class_idx"])
            valid_idx.append(idx)

        return (
            np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            pd.Index(valid_idx),
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _normalize(landmarks: np.ndarray) -> np.ndarray:
        wrist = landmarks[0]
        normalized = landmarks - wrist
        scale = np.max(np.abs(normalized)) + 1e-6
        return normalized / scale


def build_landmark_classifier(config: MediaPipeConfig) -> Model:
    inputs = Input(shape=(FEATURE_DIM,))

    x = Dense(256, activation="relu", kernel_regularizer=l2(config.l2_lambda))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout_rate)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(config.l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout_rate)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(config.l2_lambda))(x)
    x = Dropout(config.dropout_rate)(x)

    outputs = Dense(config.num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="landmark_classifier")
    model.compile(
        optimizer=Adam(config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_random_forest(config: MediaPipeConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        n_jobs=-1,
        random_state=42,
    )
