import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from models.mediapipe_classifier import LandmarkExtractor, MediaPipeConfig


@dataclass
class DemoConfig:
    model_path: Path
    class_names: list[str]
    model_type: str  # "cnn", "mobilenet", "landmark_dense", "random_forest"
    image_size: tuple[int, int] = (200, 200)
    smoothing_window: int = 10
    confidence_threshold: float = 0.8
    camera_index: int = 0


class PredictionSmoother:
    def __init__(self, window_size: int, num_classes: int):
        self._window = deque(maxlen=window_size)
        self._num_classes = num_classes

    def update(self, class_idx: int) -> int:
        self._window.append(class_idx)
        counts = np.bincount(list(self._window), minlength=self._num_classes)
        return int(np.argmax(counts))

    def reset(self) -> None:
        self._window.clear()


class BasePredictor(ABC):
    def __init__(self, config: DemoConfig):
        self.config = config

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> tuple[int, float] | tuple[None, None]:
        """Повертає (class_idx, confidence) або (None, None) якщо предикт неможливий."""

    def close(self) -> None:
        pass


class ImageModelPredictor(BasePredictor):
    def __init__(self, config: DemoConfig):
        super().__init__(config)
        self._model = load_model(str(config.model_path))

    def predict(self, frame_bgr: np.ndarray) -> tuple[int, float] | tuple[None, None]:
        img = cv2.resize(frame_bgr, self.config.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img = np.expand_dims(img, axis=0)   # (1, H, W, 1)

        probs = self._model.predict(img, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        return class_idx, confidence


class LandmarkDensePredictor(BasePredictor):
    def __init__(self, config: DemoConfig):
        super().__init__(config)
        self._model = load_model(str(config.model_path))
        self._extractor = LandmarkExtractor(MediaPipeConfig(static_image_mode=False))

    def predict(self, frame_bgr: np.ndarray) -> tuple[int, float] | tuple[None, None]:
        features = self._extractor.extract_from_frame(frame_bgr)
        if features is None:
            return None, None

        probs = self._model.predict(features[np.newaxis], verbose=0)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        return class_idx, confidence

    def close(self) -> None:
        self._extractor.close()


class RandomForestPredictor(BasePredictor):
    def __init__(self, config: DemoConfig):
        super().__init__(config)
        self._model = joblib.load(config.model_path)
        self._extractor = LandmarkExtractor(MediaPipeConfig(static_image_mode=False))

    def predict(self, frame_bgr: np.ndarray) -> tuple[int, float] | tuple[None, None]:
        features = self._extractor.extract_from_frame(frame_bgr)
        if features is None:
            return None, None

        probs = self._model.predict_proba(features[np.newaxis])[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        return class_idx, confidence

    def close(self) -> None:
        self._extractor.close()


def _build_predictor(config: DemoConfig) -> BasePredictor:
    match config.model_type:
        case "cnn" | "mobilenet":
            return ImageModelPredictor(config)
        case "landmark_dense":
            return LandmarkDensePredictor(config)
        case "random_forest":
            return RandomForestPredictor(config)
        case _:
            raise ValueError(f"Невідомий тип моделі: {config.model_type}")


class WebcamDemo:
    _WINDOW = "ASL Recognition — press Q to quit, BACKSPACE to delete"
    _MAX_OUTPUT_CHARS = 40  # скільки символів влазить в один рядок на екрані

    def __init__(self, config: DemoConfig):
        self.config = config
        self._predictor = _build_predictor(config)
        self._smoother = PredictionSmoother(config.smoothing_window, len(config.class_names))
        self._cap: cv2.VideoCapture | None = None
        self._output: str = ""
        self._last_emitted: str | None = None

    def run(self) -> None:
        self._cap = cv2.VideoCapture(self.config.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Не вдалось відкрити камеру {self.config.camera_index}")

        fps_tracker = _FPSTracker()

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                class_idx, confidence = self._predictor.predict(frame)

                if class_idx is not None and confidence >= self.config.confidence_threshold:
                    smoothed_idx = self._smoother.update(class_idx)
                    label = self.config.class_names[smoothed_idx]
                    conf_text = f"{confidence:.2f}"
                    self._emit(label)
                else:
                    label = "—"
                    conf_text = ""

                fps = fps_tracker.tick()
                _render_overlay(frame, label, conf_text, fps, self._output)
                cv2.imshow(self._WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 8:  # backspace
                    self._output = self._output[:-1]
        finally:
            self._cleanup()

    def _emit(self, label: str) -> None:
        if label in ("nothing", "???"):
            return
        if label == self._last_emitted:
            return
        self._last_emitted = label
        if label == "space":
            self._output += " "
        elif label == "del":
            self._output = self._output[:-1]
        else:
            self._output += label
        # обрізаємо щоб не виходило за межі рядка
        self._output = self._output[-self._MAX_OUTPUT_CHARS:]

    def _cleanup(self) -> None:
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self._predictor.close()


class _FPSTracker:
    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def _render_overlay(frame: np.ndarray, label: str, confidence: str,
                    fps: float, output: str) -> None:
    h, w = frame.shape[:2]

    # нижня панель — поточна буква + confidence
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
    if confidence:
        cv2.putText(frame, f"conf: {confidence}", (20, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # верхня панель — накопичений текст
    cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.putText(frame, output or "...", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


