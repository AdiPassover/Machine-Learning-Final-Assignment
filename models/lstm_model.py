from .model import BaseClassificationModel

import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.regularizers import l2

class LSTMModel(BaseClassificationModel):
    def __init__(self, input_shape, num_classes, hidden_units=128, dropout=0.3, recurrent_dropout=0.3, lr=0.001, epochs=20, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.recurrent_dropout = recurrent_dropout
        self.model = self._build_model()
        super().__init__(model=self.model)

    def _build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),  # Use Input layer here
            LSTM(self.hidden_units, recurrent_dropout=self.recurrent_dropout),
            Dropout(self.dropout),
            Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.001))
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        if X_val is not None:
            print("Using validation data for training.")
            y_val = to_categorical(y_val, num_classes=self.num_classes)
            self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val) if X_val is not None else None,
                callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                verbose=verbose
            )
        else:
            print("Validation data not provided, training without validation.")
            self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)


    def predict(self, X_test, verbose=0):
        preds = self.model.predict(X_test, verbose=verbose)
        return np.argmax(preds, axis=1)

