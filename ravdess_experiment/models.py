from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    Input,
    MaxPool2D,
    AvgPool2D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.core import Reshape

window = 55
models = {
    "A": Sequential(
        [
            Input(shape=(20, window)),
            Reshape((20, window, 1)),
            Conv2D(64, 3, activation="relu"),
            MaxPool2D((1, 2)),
            Conv2D(64, 3, activation="relu"),
            MaxPool2D((1, 2)),
            Conv2D(64, (1, 3), activation="relu"),
            MaxPool2D((1, 2)),
            Conv2D(64, (1, 3), activation="relu"),
            MaxPool2D((2, 1)),
            Conv2D(128, 3, activation="relu"),
            Dropout(0.3),
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.4),
            Dense(8, activation="softmax"),
        ]
    ),
    "B": Sequential(
        [
            Input(shape=(20, window)),
            Reshape((20, window, 1)),
            Conv2D(64, 3, activation="relu"),
            MaxPool2D((1, 2)),
            Conv2D(64, 3, activation="relu"),
            MaxPool2D((1, 2)),
            Conv2D(128, 3, activation="relu"),
            MaxPool2D((1, 2)),
            MaxPool2D((2, 1)),
            Conv2D(128, 4, activation="relu"),
            Dropout(0.3),
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.4),
            Dense(8, activation="softmax"),
        ]
    ),
    "C": Sequential(
        [
            Input(shape=(20, window)),
            Reshape((20, window, 1)),
            Conv2D(64, 3, activation="relu"),
            MaxPool2D((1, 2)),
            Conv2D(64, (1, 3), activation="relu"),
            MaxPool2D((1, 2)),
            Conv2D(64, (1, 3), activation="relu"),
            MaxPool2D((2, 1)),
            Conv2D(128, 3, activation="relu"),
            Dropout(0.3),
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.4),
            Dense(1024, activation="relu"),
            Dropout(0.4),
            Dense(8, activation="softmax"),
        ]
    ),
}


def get_model(model_name):
    try:
        model = load_model(model_name + ".h5")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())
        return model
    except:
        model = models[model_name]
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())
        model.save(model_name + ".h5")
    return model


from tensorflow.keras.utils import plot_model

modelitos = ["A", "B", "C"]
for m in modelitos:
    plot_model(
        get_model(m), to_file=m + ".png", show_shapes=True, show_layer_names=False
    )
