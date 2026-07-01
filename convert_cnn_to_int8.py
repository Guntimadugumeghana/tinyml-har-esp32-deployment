"""
Convert har_cnn_model.keras to a full INT8 TFLite model (int8 in, int8 out).

Run this in your repo folder:
    cd /mnt/DATA/SPLAB/tinymlhar/
    CUDA_VISIBLE_DEVICES=-1 python3 convert_cnn_to_int8.py

Requires X_test.npy to exist (used as the calibration/representative dataset).
Produces: har_cnn_int8.tflite
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

MODEL_PATH = "har_cnn_model.keras"
X_TEST_PATH = "X_test.npy"
OUTPUT_PATH = "har_cnn_int8.tflite"
N_CALIBRATION_SAMPLES = 300


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found in current directory")
    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(f"{X_TEST_PATH} not found - needed for calibration")

    model = tf.keras.models.load_model(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)

    def representative_dataset():
        for i in range(min(N_CALIBRATION_SAMPLES, len(X_test))):
            yield [X_test[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Wrote {OUTPUT_PATH}: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)")

    # Confirm no Flex/Select ops crept in
    interp = tf.lite.Interpreter(model_path=OUTPUT_PATH)
    interp.allocate_tensors()
    ops_used = sorted(set(op["op_name"] for op in interp._get_ops_details()))
    print("Ops used:", ops_used)
    if any("FLEX" in op.upper() for op in ops_used):
        print("WARNING: Flex op detected - this will NOT run on TFLite Micro")
    else:
        print("No Flex ops - should be TFLite Micro compatible")


if __name__ == "__main__":
    main()