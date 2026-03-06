from dataclasses import dataclass

from tensorflow.keras import Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


@dataclass
class MobileNetConfig:
    input_shape: tuple[int, int, int] = (200, 200, 1)
    num_classes: int = 29
    dropout_rate: float = 0.4
    l2_lambda: float = 1e-4
    learning_rate_head: float = 3e-4
    learning_rate_finetune: float = 1e-5
    unfreeze_from_layer: int = 100  # шари починаючи з якого розморожуємо при fine-tune


def build_mobilenet(config: MobileNetConfig) -> Model:
    base = MobileNetV2(
        input_shape=(config.input_shape[0], config.input_shape[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = Input(shape=config.input_shape)
    x = Concatenate()([inputs, inputs, inputs])  # (H, W, 1) → (H, W, 3)
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(config.l2_lambda))(x)
    x = Dropout(config.dropout_rate)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(config.l2_lambda))(x)
    x = Dropout(config.dropout_rate)(x)
    outputs = Dense(config.num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="mobilenet_transfer")
    _compile(model, config.learning_rate_head)
    return model


def unfreeze_for_finetuning(model: Model, config: MobileNetConfig) -> None:
    base = model.get_layer("mobilenetv2_1.00_224")
    for layer in base.layers[:config.unfreeze_from_layer]:
        layer.trainable = False
    for layer in base.layers[config.unfreeze_from_layer:]:
        layer.trainable = True
    _compile(model, config.learning_rate_finetune)


def _compile(model: Model, learning_rate: float) -> None:
    model.compile(
        optimizer=Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
