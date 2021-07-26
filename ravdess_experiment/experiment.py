import numpy as np
from models import get_model
from gen_dataset import get_dataset
import datetime
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


## This file just is a test for verifyng experiment 

dataset = get_dataset(cache=True)
train_x, train_y, test_x, test_y = dataset

print(train_x.shape)
print(set(list(train_y)))


model = get_model("A")
model.summary()

log_dir = "logs/fit/A_5"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_x,
    train_y,
    validation_data=(test_x, test_y),
    epochs=200,
    batch_size=64,
    callbacks=[
        tensorboard_callback,
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20),
    ],
)


y_out = np.argmax(model.predict(test_x), axis=1)

print(confusion_matrix(test_y, y_out))
report = classification_report(test_y, y_out)
print(report)
