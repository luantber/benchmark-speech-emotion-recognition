from tensorflow.python.keras import models
from models import get_model
from gen_dataset import get_dataset, split_set
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

import gc


def train(model_name, x_train, y_train, x_test, y_test):
    log_dir = "logs/fit/" + model_name

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir, histogram_freq=1
    # )

    model = get_model(model_name)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=150,
        batch_size=256,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=15),
            # tensorboard_callback,
        ],
        verbose=0,
    )
    report = classification_report(
        y_test, np.argmax(model.predict(x_test), axis=1), output_dict=True
    )

    del model
    gc.collect()
    tf.keras.backend.clear_session()
    return report["weighted avg"]["f1-score"]


def print_matrix(matrix):
    print("__________")
    for m in matrix:
        print(m)
    print("__________")


iterations = 30
modelos = ["A", "B", "C"]


results_matrix = np.zeros((len(modelos), iterations, 5))

for i in range(iterations):
    dataset = get_dataset(cache=True)
    kf = KFold(n_splits=5, shuffle=True)
    j = 0
    for train_index, test_index in kf.split(dataset):
        t1 = time.time()

        print(j)
        x_train, y_train = split_set([dataset[xi] for xi in train_index])
        x_test, y_test = split_set([dataset[xi] for xi in test_index])

        for m_i in range(len(modelos)):
            result = train(modelos[m_i], x_train, y_train, x_test, y_test)
            results_matrix[m_i][i][j] = result
            print("o")

        j += 1
        print(time.time() - t1)
    print(results_matrix[:, i, :])
    np.save("matrix_3", results_matrix)

print_matrix(results_matrix)
