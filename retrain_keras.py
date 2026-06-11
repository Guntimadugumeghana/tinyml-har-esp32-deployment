"""
Step 1: Retrain HAR LSTM in Keras (TF2)
Rebuilds the original TF1 LSTM architecture in modern Keras,
trains on UCI HAR dataset, saves as .keras format for TFLite conversion.

Architecture mirrors the original:
  - Input: (128 timesteps, 9 features)
  - 2 stacked LSTM layers (32 units each)
  - Dense softmax output (6 classes)
"""

import numpy as np
import os
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics

# Reproducibility 
tf.random.set_seed(42)
np.random.seed(42)

# Constants (match original repo) 
INPUT_SIGNAL_TYPES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_",
]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]

N_STEPS     = 128   # timesteps per window
N_FEATURES  = 9     # sensor channels
N_CLASSES   = 6
N_HIDDEN    = 32    # LSTM units (kept small for MCU fit)
BATCH_SIZE  = 128
EPOCHS      = 30
DATASET_DIR = "/mnt/DATA/SPLAB/tinymlhar/UCI_HAR_Dataset"

def download_dataset():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00240/UCI%20HAR%20Dataset.zip")
    zip_path = "/mnt/DATA/SPLAB/tinymlhar/UCI_HAR.zip"
    if not os.path.exists(DATASET_DIR):
        print("Downloading UCI HAR Dataset (~60 MB)…")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/mnt/DATA/SPLAB/tinymlhar/")
        os.rename("/mnt/DATA/SPLAB/tinymlhar/UCI HAR Dataset", DATASET_DIR)
        os.remove(zip_path)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already present.")

# Load X signals 
def load_X(split):
    """Load all 9 sensor channels for a split (train/test)."""
    signals = []
    for sig in INPUT_SIGNAL_TYPES:
        path = os.path.join(DATASET_DIR, split, "Inertial Signals",
                            f"{sig}{split}.txt")
        data = np.loadtxt(path)          # shape: (N, 128)
        signals.append(data)
    # Stack → (N, 128, 9)
    return np.stack(signals, axis=-1).astype(np.float32)

# ── Load y labels ─────────────────────────────────────────────────────────────
def load_y(split):
    path = os.path.join(DATASET_DIR, split, f"y_{split}.txt")
    return np.loadtxt(path, dtype=np.int32) - 1  # 0-based

# ── Build Keras model ─────────────────────────────────────────────────────────
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(N_STEPS, N_FEATURES), name="accel_input"),
        keras.layers.LSTM(
            N_HIDDEN,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0.0,
            unroll=False,
            use_bias=True,
            name="lstm_1"
        ),

        keras.layers.LSTM(
            N_HIDDEN,
            return_sequences=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0.0,
            unroll=False,
            use_bias=True,
            name="lstm_2"
        ),
        keras.layers.Dense(N_CLASSES, activation="softmax", name="output"),
    ], name="HAR_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load pre-saved data (either from UCI HAR or synthetic equivalent)
    print("\nLoading data…")
    if os.path.exists("/mnt/DATA/SPLAB/tinymlhar/X_train.npy"):
        X_train = np.load("/mnt/DATA/SPLAB/tinymlhar/X_train.npy")
        y_train = np.load("/mnt/DATA/SPLAB/tinymlhar/y_train.npy")
        X_test  = np.load("/mnt/DATA/SPLAB/tinymlhar/X_test.npy")
        y_test  = np.load("/mnt/DATA/SPLAB/tinymlhar/y_test.npy")
    else:
        download_dataset()
        X_train = load_X("train")
        y_train = load_y("train")
        X_test  = load_X("test")
        y_test  = load_y("test")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                       monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    print("\nTraining…")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(f"\n{'='*50}")
    print(f"  Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss           : {test_loss:.4f}")
    print(f"  Precision           : "
          f"{metrics.precision_score(y_test, predictions, average='weighted')*100:.2f}%")
    print(f"  Recall              : "
          f"{metrics.recall_score(y_test, predictions, average='weighted')*100:.2f}%")
    print(f"  F1-Score            : "
          f"{metrics.f1_score(y_test, predictions, average='weighted')*100:.2f}%")
    print(f"{'='*50}")

    print("\nConfusion Matrix:")
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)
    print("\nClass labels:", LABELS)

    # ── Save model ────────────────────────────────────────────────────────────
    save_path = "/mnt/DATA/SPLAB/tinymlhar/har_model.keras"
    model.save(save_path)
    print(f"\nModel saved → {save_path}")

    # Also save test data for validation script
    np.save("/mnt/DATA/SPLAB/tinymlhar/X_test.npy", X_test)
    np.save("/mnt/DATA/SPLAB/tinymlhar/y_test.npy", y_test)
    print("Test data saved → /mnt/DATA/SPLAB/tinymlhar/X_test.npy, y_test.npy")

if __name__ == "__main__":
    main()