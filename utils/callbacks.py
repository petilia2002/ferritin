from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def early_stopping() -> EarlyStopping:
    return EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=100,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
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


__all__ = ["early_stopping", "reduce_lr"]
