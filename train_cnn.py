"""
Train a 1D-CNN for HAR on the UCI HAR dataset (LSTM -> CNN pivot).

Why: the existing LSTM (1_retrain_keras.py) produces a Keras model that
TFLite converts using FlexTensorListReserve (a Select-TF-Ops / Flex op).
That op has no TFLite Micro kernel, so the model cannot run on bare-metal
ESP32 at all. Conv1D/MaxPool1D/Dense have no such requirement -- they lower
to CONV_2D / MAX_POOL_2D / FULLY_CONNECTED / MEAN / SOFTMAX, all of which
ship in TFLite Micro's standard op set.

Architecture: Conv1D -> MaxPool1D -> Conv1D -> MaxPool1D -> Conv1D ->
GlobalAveragePooling1D -> Dense(6, softmax). GlobalAveragePooling1D is used
instead of Flatten+Dense specifically to keep parameter count (and final
flash footprint) small -- Flatten would balloon the first Dense layer's
weight count by 32x.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU, matches existing scripts

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

tf.random.set_seed(42)
np.random.seed(42)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(REPO_DIR, "UCI_HAR_Dataset")

INPUT_SIGNAL_TYPES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_",
]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]

N_STEPS, N_FEATURES, N_CLASSES = 128, 9, 6
BATCH_SIZE, EPOCHS = 64, 40


def load_X(split):
    signals = []
    for sig in INPUT_SIGNAL_TYPES:
        path = os.path.join(DATASET_DIR, split, "Inertial Signals", f"{sig}{split}.txt")
        signals.append(np.loadtxt(path))
    return np.stack(signals, axis=-1).astype(np.float32)


def load_y(split):
    path = os.path.join(DATASET_DIR, split, f"y_{split}.txt")
    return np.loadtxt(path, dtype=np.int32) - 1


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(N_STEPS, N_FEATURES), name="accel_input"),
        keras.layers.Conv1D(16, 5, padding="same", activation="relu", name="conv1"),
        keras.layers.MaxPooling1D(2, name="pool1"),                # 128 -> 64
        keras.layers.Conv1D(32, 5, padding="same", activation="relu", name="conv2"),
        keras.layers.MaxPooling1D(2, name="pool2"),                # 64 -> 32
        keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv3"),
        keras.layers.GlobalAveragePooling1D(name="gap"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(N_CLASSES, activation="softmax", name="output"),
    ], name="HAR_CNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading data from raw UCI HAR text files...")
    X_train, y_train = load_X("train"), load_y("train")
    X_test, y_test = load_X("test"), load_y("test")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    # Save for reuse by the conversion/validation scripts (mirrors existing repo pattern)
    np.save(os.path.join(REPO_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(REPO_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(REPO_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(REPO_DIR, "y_test.npy"), y_test)

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(f"\n{'='*50}")
    print(f"  Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss           : {test_loss:.4f}")
    print(f"  Precision           : {metrics.precision_score(y_test, predictions, average='weighted')*100:.2f}%")
    print(f"  Recall              : {metrics.recall_score(y_test, predictions, average='weighted')*100:.2f}%")
    print(f"  F1-Score            : {metrics.f1_score(y_test, predictions, average='weighted')*100:.2f}%")
    print(f"{'='*50}")

    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix(y_test, predictions))
    print("\nPer-class accuracy:")
    cm = metrics.confusion_matrix(y_test, predictions)
    for i, label in enumerate(LABELS):
        print(f"  {label:20s}: {cm[i, i] / cm[i].sum() * 100:.1f}%")

    save_path = os.path.join(REPO_DIR, "har_cnn_model.keras")
    model.save(save_path)
    print(f"\nModel saved -> {save_path}")
    print(f"Total params: {model.count_params()}")


if __name__ == "__main__":
    main()