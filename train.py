import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras import Input, datasets
from keras.layers import Dense
from keras.models import Sequential


def load_data(single_batch=False):
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    if single_batch:
        return X_train[:32], y_train[:32], X_test, y_test

    return X_train, y_train, X_test, y_test


def build_model():
    """Build a simple neural network model using
    the supplied output layer."""

    model = Sequential(
        [
            Input(shape=(784,)),
            Dense(512, activation="relu"),
            Dense(256, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    return model


def fit(single_batch=False):
    model = build_model()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    X_train, y_train, X_test, y_test = load_data(single_batch)

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=20,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    return model, history
