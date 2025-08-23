from .model import BaseClassificationModel

import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.optimizers import Adam


class CNNModel(BaseClassificationModel):
    def __init__(self, input_shape, num_classes,
                 dropout=0.3, lr=0.001, epochs=20, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        super().__init__(model=self.model)

    def _build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),  # (F, T, 1)

            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(self.dropout),
            Dense(self.num_classes, activation="softmax")
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        if X_val is not None:
            y_val = to_categorical(y_val, num_classes=self.num_classes)
            self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                verbose=verbose
            )
        else:
            self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=verbose
            )

    def predict(self, X_test, verbose=0):
        preds = self.model.predict(X_test, verbose=verbose)
        return np.argmax(preds, axis=1)
