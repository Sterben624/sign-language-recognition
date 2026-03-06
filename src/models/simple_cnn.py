from dataclasses import dataclass

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


@dataclass
class CNNConfig:
    input_shape: tuple[int, int, int] = (200, 200, 1)
    num_classes: int = 29
    dropout_rate: float = 0.4
    l2_lambda: float = 1e-4
    learning_rate: float = 1e-3


def build_simple_cnn(config: CNNConfig) -> Model:
    inputs = Input(shape=config.input_shape)

    x = _conv_block(inputs, filters=32, config=config)
    x = _conv_block(x, filters=64, config=config)
    x = _conv_block(x, filters=128, config=config)
    x = _conv_block(x, filters=256, config=config)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu", kernel_regularizer=l2(config.l2_lambda))(x)
    x = Dropout(config.dropout_rate)(x)
    outputs = Dense(config.num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="simple_cnn")
    model.compile(
        optimizer=Adam(config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _conv_block(x, filters: int, config: CNNConfig):
    x = Conv2D(filters, (3, 3), padding="same", activation="relu",
               kernel_regularizer=l2(config.l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    return x
