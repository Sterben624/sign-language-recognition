from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


@dataclass
class DataConfig:
    data_dir: Path
    image_size: tuple[int, int] = (200, 200)
    batch_size: int = 32
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    grayscale: bool = True


class ASLDataLoader:
    TRAIN_SUBDIR = "asl_alphabet_train"

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dir = Path(config.data_dir) / self.TRAIN_SUBDIR
        self._df: pd.DataFrame | None = None
        self.classes: list[str] | None = None
        self.class_to_idx: dict[str, int] | None = None

    @property
    def num_classes(self) -> int:
        return len(self.classes) if self.classes else 0

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._scan_dataset()
        return self._df

    def _scan_dataset(self) -> pd.DataFrame:
        class_dirs = sorted(d for d in self.train_dir.iterdir() if d.is_dir())
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        records = [
            {"filepath": str(img), "label": cls_dir.name, "class_idx": self.class_to_idx[cls_dir.name]}
            for cls_dir in class_dirs
            for img in cls_dir.glob("*.jpg")
        ]
        return pd.DataFrame(records)

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        val_test_size = self.config.val_split + self.config.test_split
        relative_test = self.config.test_split / val_test_size

        train_df, temp_df = train_test_split(
            self.df,
            test_size=val_test_size,
            stratify=self.df["label"],
            random_state=self.config.seed,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            stratify=temp_df["label"],
            random_state=self.config.seed,
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def image_generator(self, df: pd.DataFrame, augment: bool = False):
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            **(self._aug_args() if augment else {}),
        )
        return datagen.flow_from_dataframe(
            dataframe=df,
            x_col="filepath",
            y_col="label",
            directory=None,
            target_size=self.config.image_size,
            color_mode="grayscale" if self.config.grayscale else "rgb",
            batch_size=self.config.batch_size,
            class_mode="categorical",
            classes=self.classes,
            shuffle=augment,
            seed=self.config.seed,
        )

    def load_images_into_memory(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        images, labels = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
            img = cv2.imread(row["filepath"])
            if self.config.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, self.config.image_size)
                img = np.expand_dims(img, axis=-1)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.config.image_size)
            images.append(img / 255.0)
            labels.append(row["class_idx"])
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    @staticmethod
    def _aug_args() -> dict:
        return {
            "rotation_range": 10,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.1,
            "horizontal_flip": False,  # ASL letters are not symmetric
        }
