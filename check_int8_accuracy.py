"""
Check accuracy of har_cnn_int8.tflite on X_test.npy / y_test.npy.

Run in your repo folder:
    cd /mnt/DATA/SPLAB/tinymlhar/
    python3 check_int8_accuracy.py
"""

import numpy as np
import tensorflow as tf

MODEL_PATH = "har_cnn_int8.tflite"
X_TEST_PATH = "X_test.npy"
Y_TEST_PATH = "y_test.npy"


def main():
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zero_point = inp["quantization"]

    correct = 0
    preds = np.zeros(len(X_test), dtype=np.int32)
    for i in range(len(X_test)):
        x = X_test[i:i + 1]
        x_q = (x / scale + zero_point).astype(np.int8)
        interp.set_tensor(inp["index"], x_q)
        interp.invoke()
        pred = np.argmax(interp.get_tensor(out["index"])[0])
        preds[i] = pred
        correct += int(pred == y_test[i])

    acc = correct / len(X_test) * 100
    print(f"INT8 accuracy: {acc:.2f}%  ({correct}/{len(X_test)})")
    print(f"Input quant: scale={scale}, zero_point={zero_point}")

    labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]
    print("\nPer-class accuracy:")
    for c in range(6):
        mask = (y_test == c)
        class_acc = (preds[mask] == c).sum() / mask.sum() * 100
        print(f"  {labels[c]:20s}: {class_acc:.1f}%")


if __name__ == "__main__":
    main()