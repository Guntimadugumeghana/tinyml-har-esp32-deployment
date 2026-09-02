# TinyML Human Activity Recognition on ESP32

End-to-end deployment of a quantized 1D-CNN model for real-time Human Activity Recognition on a simulated ESP32 microcontroller, using TensorFlow Lite Micro.

## Overview

This project demonstrates the full TinyML pipeline — training, quantization, and embedded deployment - on the UCI HAR dataset, targeting a simulated ESP32 (Wokwi) with TFLite Micro.

The original architecture was an LSTM. It was replaced with a 1D-CNN after the LSTM proved fundamentally incompatible with TFLite Micro on bare-metal hardware. That pivot, why it was necessary, and what it cost/gained, is documented below rather than glossed over.

## Why the architecture changed: LSTM → 1D-CNN

A standard Keras LSTM was trained first (16/32/64-unit variants, 83.5–88.7% accuracy). All of them convert to TFLite using a `FlexTensorListReserve` op — part of TensorFlow's Select/Flex op set. TFLite Micro, which is what actually runs on bare-metal ESP32, does not implement Flex ops. There is no workaround inside the LSTM architecture itself: a plain `tf.lite.Interpreter` on this machine fails immediately on `AllocateTensors()` with:

```
RuntimeError: Select TensorFlow op(s), included in the given model, is(are) not
supported by this interpreter... Node number 1 (FlexTensorListReserve) failed to prepare.
```

Using `unroll=True` removes the Flex op but introduces a different blocker: a `FILL` kernel failure (`Non-constant dims tensor not supported`), which is a TFLite Micro kernel limitation, not a configuration issue.

The fix was switching to a 1D-CNN. Conv1D/MaxPool1D/GlobalAveragePooling1D lower to `CONV_2D`, `MAX_POOL_2D`, `MEAN`, `FULLY_CONNECTED`, `SOFTMAX`, `RESHAPE`, `EXPAND_DIMS` — all standard TFLite Micro ops, zero Flex/Select ops required. This is also the standard industry approach for bare-metal HAR; LSTMs are well known to be a poor fit for microcontroller deployment for exactly this reason.

The CNN ended up smaller and more accurate than the largest LSTM tried.

## Results

| Metric | LSTM (32-unit) | 1D-CNN (this project) |
|---|---|---|
| Float32 accuracy | 88.73% | 91.3% ± 1.1% (range 89.6–92.7%, 5 runs) |
| INT8 accuracy | 88.73% (dynamic-range only — see note) | 91.48% |
| Accuracy drop from INT8 | 0.00% | ~0% (within run-to-run noise) |
| Parameters | 13,894 | 9,926 |
| Float32 TFLite size | 73.0 KB | 45 KB |
| INT8 TFLite size | 39.8 KB (dynamic-range) | 19.92 KB (full INT8) |
| TFLite Micro compatible | No (Flex ops) | Yes (0 Flex ops) |
| Tensor arena used | N/A (never ran on-device) | 6.93 KB |
| Wokwi inference latency | N/A | ~592 ms |

**Note on accuracy variance:** training the CNN is not bit-reproducible across runs even with a fixed random seed — TensorFlow's oneDNN backend reorders floating-point operations differently depending on CPU instruction set, which changes early gradients and compounds over training. Five identical training runs on the same machine produced 89.62%, 92.67%, 91.21%, 91.55%, 91.45% (mean 91.3%, stdev 1.1%). The model used for INT8 conversion and deployment below is the 91.45% run.

**Note on LSTM INT8 row:** the LSTM's "INT8" model uses dynamic-range quantization (weights int8, activations float at runtime) because full INT8 calibration isn't possible with the Flex ops present. It was also never deployed to TFLite Micro — the comparison row is float32-vs-quantized-on-desktop only. It never ran on the ESP32 simulation at all, because it can't.

### Per-class accuracy (1D-CNN, INT8, on-device representative run)

| Class | Accuracy |
|---|---|
| WALKING | 99.2% |
| WALKING_UPSTAIRS | 93.8% |
| WALKING_DOWNSTAIRS | 98.8% |
| SITTING | 82.1% |
| STANDING | 81.6% |
| LAYING | 95.0% |

SITTING and STANDING are the most-confused pair, consistent with the underlying IMU signal — these two postures look nearly identical to an accelerometer/gyroscope. This shows up directly in the live Wokwi run below (one SITTING fixture predicted as STANDING at 81.2% confidence).

## Model architecture

```
Input: (128 timesteps, 9 features)
  └─ Conv1D(16, kernel=5, padding=same, relu)
  └─ MaxPooling1D(2)                          # 128 -> 64
  └─ Conv1D(32, kernel=5, padding=same, relu)
  └─ MaxPooling1D(2)                          # 64 -> 32
  └─ Conv1D(64, kernel=3, padding=same, relu)
  └─ GlobalAveragePooling1D
  └─ Dropout(0.3)
  └─ Dense(6, softmax)

Total trainable parameters: 9,926
```

GlobalAveragePooling1D is used instead of Flatten+Dense specifically to keep the parameter count down — Flatten would have added roughly 130K parameters to the first dense layer for negligible accuracy gain.

9 sensor channels: `body_acc_x/y/z`, `body_gyro_x/y/z`, `total_acc_x/y/z`
6 output classes: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`

## Verified on-device behavior (Wokwi ESP32 simulation)

The compiled sketch was run on a simulated ESP32 (`board-esp32-devkit-c-v4`) via `wokwi-cli`, with real serial output captured to a log file. It cycles through one real UCI HAR test-set sample per class:

```
Arena used: 6.93 KB
Input tensor type: INT8
Cycling through 6 activity fixtures...
True: WALKING              | Predicted: WALKING              (99.6%)  Latency: 592.40 ms
True: WALKING_UPSTAIRS      | Predicted: WALKING_UPSTAIRS      (99.6%)  Latency: 592.36 ms
True: WALKING_DOWNSTAIRS    | Predicted: WALKING_DOWNSTAIRS    (99.6%)  Latency: 592.40 ms
True: SITTING               | Predicted: STANDING              (81.2%)  Latency: 592.56 ms
True: STANDING              | Predicted: STANDING              (99.6%)  Latency: 592.56 ms
True: LAYING                | Predicted: LAYING                (99.6%)  Latency: 592.49 ms
```

5 of 6 fixtures predicted correctly. The SITTING misclassification is the same confusion seen in the held-out test set, not a deployment bug.

**Important scope note:** this is a Wokwi simulation, not a physical ESP32. There is no real MPU-6050 sensor — inference runs on fixed test-set samples (`har_fixtures.h`) cycled in a loop, not live sensor data. The MPU-6050 I2C driver code in `tinymlhar.ino` is present and wired in `diagram.json` but currently unused by `loop()`.

## ESP32 resource usage

| Resource | Usage |
|---|---|
| Flash | 429,704 bytes / 1,310,720 bytes (32%) |
| RAM (global variables) | 65,036 bytes / 327,680 bytes (19%) |
| Tensor arena (allocated) | 12 KB |
| Tensor arena (actually used) | 6.93 KB |

## Pipeline

```
UCI HAR Dataset
      │
      ▼
1. Train 1D-CNN in Keras (TF2)              [train_cnn.py]
      │  91.3% ± 1.1% accuracy (5-run range), input: (128, 9)
      ▼
2. Convert to full INT8 TFLite              [convert_cnn_to_int8.py]
      │  float32: 45 KB → INT8: 19.92 KB, zero Flex ops
      ▼
3. Validate INT8 model accuracy             [check_int8_accuracy.py]
      │  91.48%, ~0% drop from float32
      ▼
4. Generate C header                        [generate_model_header.py]
      │  har_model_data.h — int8 byte array, 20,400 bytes
      ▼
5. Generate on-device test fixtures         [generate_fixtures.py]
      │  har_fixtures.h — one real test sample per class
      ▼
6. Deploy on ESP32 (Arduino C++ / TFLite Micro)   [tinymlhar.ino]
      │  arduino-cli compile, wokwi-cli simulate
      ▼
Verified inference @ ~592 ms/sample, 6.93 KB arena
```

## Project structure

```
tinyml-har-esp32-deployment/
├── train_cnn.py                # Train 1D-CNN on UCI HAR, save .keras model
├── convert_cnn_to_int8.py      # Convert to full INT8 TFLite (calibrated)
├── check_int8_accuracy.py      # Validate INT8 model accuracy on test set
├── generate_model_header.py    # har_cnn_int8.tflite -> har_model_data.h
├── generate_fixtures.py        # Real test samples -> har_fixtures.h
├── tinymlhar.ino                # ESP32 Arduino sketch (TFLite Micro inference)
├── har_model_data.h             # Auto-generated C array of quantized model
├── har_fixtures.h               # Auto-generated test fixtures (real samples)
├── diagram.json                 # Wokwi ESP32 board config
├── wokwi.toml                   # Wokwi simulator config
├── archive/                     # Earlier LSTM attempt — kept for the pivot story
│   ├── 1_retrain_keras.py
│   ├── 2_convert_to_tflite.py
│   ├── 3_validate_tflite.py
│   ├── har_lstm16.keras / har_lstm16_quant.tflite
│   ├── har_lstm32.keras / har_lstm32_quant.tflite
│   └── har_lstm64.keras / har_lstm64_quant.tflite
└── README.md
```

## Reproduce

Requirements:
```bash
pip install tensorflow scikit-learn numpy
```

Step 1 — Train the CNN:
```bash
CUDA_VISIBLE_DEVICES=-1 python3 train_cnn.py
```
Note: training is not bit-reproducible run to run (see accuracy variance note above). Expect 89–93%.

Step 2 — Convert to INT8:
```bash
CUDA_VISIBLE_DEVICES=-1 python3 convert_cnn_to_int8.py
```
`CUDA_VISIBLE_DEVICES=-1` forces CPU mode — without it, TensorFlow may use a GPU op incompatible with TFLite conversion.

Step 3 — Validate INT8 accuracy:
```bash
python3 check_int8_accuracy.py
```

Step 4 — Generate the model header and fixtures:
```bash
python3 generate_model_header.py
python3 generate_fixtures.py
```

Step 5 — Compile and simulate:
```bash
arduino-cli compile --fqbn esp32:esp32:esp32 --build-path ./build --clean .
wokwi-cli --serial-log-file serial.log .
```
Requires a Wokwi CLI token from https://wokwi.com/dashboard/ci, set as `WOKWI_CLI_TOKEN`.

## Tech stack

TensorFlow · TensorFlow Lite · Keras · TFLite Micro · ESP32 · Arduino C++ · Wokwi · arduino-cli · Python · scikit-learn

## Dataset

UCI Human Activity Recognition Using Smartphones — 30 subjects, Samsung Galaxy S2, 50 Hz sampling rate, 128-sample sliding windows, 9 raw inertial signal channels, 6 activity classes.

## Author

Guntimadugumeghana
