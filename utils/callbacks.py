import numpy as np
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


class AverageWeights(Callback):
    def __init__(self, window: int = 5):
        super().__init__()
        self.window = window
        self.last_weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.last_weights.append(self.model.get_weights())
        self.last_weights = (
            self.last_weights[-self.window :]
            if len(self.last_weights) > self.window
            else self.last_weights
        )

    def on_train_end(self, logs=None):
        average_weights = []
        for weights in zip(*self.last_weights):
            average_weights.append(np.mean(weights, axis=0))
        self.model.set_weights(average_weights)
        print(f"Setting weights over the last {self.window} epochs")


def early_stopping() -> EarlyStopping:
    return EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=100,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    )


def reduce_lr() -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=1e-6,
    )


def average_weights(window: int = 5) -> AverageWeights:
    return AverageWeights(window)


__all__ = ["early_stopping", "reduce_lr", "average_weights"]
