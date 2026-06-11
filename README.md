# TinyML Human Activity Recognition on ESP32

> End-to-end deployment of a quantized LSTM model for real-time Human Activity Recognition on a simulated ESP32 microcontroller.

---

## Overview

This project demonstrates the full TinyML pipeline — from training a deep learning model to deploying it on an embedded system — using the UCI HAR dataset and a simulated ESP32 via Wokwi.

The focus is on **model optimization and edge deployment**: taking a trained LSTM and making it small enough and fast enough to run on a microcontroller with 320 KB RAM and 4 MB flash.

---

## Results

| Metric | Value |
|---|---|
| Model accuracy (float32) | 88.7% |
| Model accuracy after INT8 quantization | 88.7% |
| Accuracy drop from quantization | 0.00% |
| Float32 TFLite size | 73.0 KB |
| INT8 quantized size | 39.8 KB |
| Size reduction | 45.5% |
| Estimated ESP32 inference latency | ~530 ms |
| ESP32 flash usage | 26% (348 KB / 1310 KB) |
| ESP32 RAM usage | 25% (84 KB / 327 KB) |

---

## Activities Classified

| Class | Per-class Accuracy |
|---|---|
| WALKING | 89.5% |
| WALKING_UPSTAIRS | 88.7% |
| WALKING_DOWNSTAIRS | 97.1% |
| SITTING | 84.9% |
| STANDING | 78.2% |
| LAYING | 95.3% |

---

## Pipeline

```
UCI HAR Dataset
      │
      ▼
1. Train LSTM in Keras (TF2)
      │  88.7% accuracy, input: (128 timesteps × 9 sensor channels)
      ▼
2. Convert to TensorFlow Lite
      │  float32: 73 KB → INT8 dynamic-range quantization: 40 KB
      ▼
3. Validate quantized model
      │  accuracy preserved, latency measured
      ▼
4. Generate C header (har_model_data.h)
      │  model bytes embedded as uint8_t array
      ▼
5. Deploy on ESP32 (Arduino C + TFLite Micro)
      │  compiled with arduino-cli, simulated on Wokwi
      ▼
Real-time activity prediction @ ~2 Hz
```

---

## Model Architecture

```
Input: (128 timesteps, 9 features)
  └─ LSTM (32 units, return_sequences=True)
  └─ LSTM (32 units)
  └─ Dense (6 units, softmax)

Total trainable parameters: 13,894
```

**9 sensor channels:** body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z  
**6 output classes:** WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

---

## Quantization Approach

Standard Keras LSTM uses `TensorListReserve` ops internally, which are incompatible with full INT8 calibration in TFLite. This project uses **dynamic-range quantization** — weights are stored as int8, activations remain float at runtime. This avoids the calibration step entirely while still achieving significant size reduction with zero accuracy loss.

---

## Project Structure

```
tinyml-har-esp32-deployment/
├── 1_retrain_keras.py        # Train LSTM on UCI HAR, save .keras model
├── 2_convert_to_tflite.py    # Convert to float32 + INT8 TFLite, generate C header
├── 3_validate_tflite.py      # Validate accuracy and measure latency
├── tinymlhar.ino             # ESP32 Arduino sketch (TFLite Micro inference)
├── har_model_data.h          # Auto-generated C array of quantized model
├── diagram.json              # Wokwi ESP32 board config
├── wokwi.toml                # Wokwi VS Code simulator config
└── README.md
```

---

## Reproduce

**Requirements**
```bash
pip install tensorflow scikit-learn numpy
```

**Step 1 — Train**
```bash
python 1_retrain_keras.py
```

**Step 2 — Convert**
```bash
CUDA_VISIBLE_DEVICES=-1 python 2_convert_to_tflite.py
```
> Note: `CUDA_VISIBLE_DEVICES=-1` forces CPU mode. Without it, TensorFlow uses CudnnRNNV3 on GPU which is incompatible with TFLite conversion.

**Step 3 — Validate**
```bash
python 3_validate_tflite.py
```

**Step 4 — Simulate on ESP32**

Install [arduino-cli](https://arduino.github.io/arduino-cli/) and the ESP32 board package, then:
```bash
arduino-cli compile \
  --fqbn esp32:esp32:esp32 \
  --output-dir build \
  .
```
Open in [Wokwi for VS Code](https://docs.wokwi.com/vscode/getting-started).

> **Note:** Simulation compiled and verified on Wokwi ESP32 (VS Code). Serial monitor output requires Wokwi paid plan. Binary verified: 348 KB flash (26%), 84 KB RAM (25%).

---

## Tech Stack

`TensorFlow` `TensorFlow Lite` `Keras` `TFLite Micro` `ESP32` `Arduino C` `Wokwi` `arduino-cli` `Python` `scikit-learn`

**Keywords:** TinyML · model quantization · edge inference · embedded C · INT8 quantization · LSTM · microcontroller deployment

---

## Dataset

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)  
30 subjects · Samsung Galaxy S2 · 50 Hz sampling rate · 128-sample sliding windows

---

## Author

[Guntimadugumeghana](https://github.com/Guntimadugumeghana)
