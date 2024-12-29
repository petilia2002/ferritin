from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=50,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.8,
    patience=10,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-6,
)

__all__ = ["early_stopping", "reduce_lr"]
