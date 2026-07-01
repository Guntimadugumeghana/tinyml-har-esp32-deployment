"""
Generate har_model_data.h from har_cnn_int8.tflite as a C byte array.

Run in your repo folder:
    cd /mnt/DATA/SPLAB/tinymlhar/
    python3 generate_model_header.py
"""

MODEL_PATH = "har_cnn_int8.tflite"
OUTPUT_PATH = "har_model_data.h"
ARRAY_NAME = "g_har_cnn_model_data"


def main():
    with open(MODEL_PATH, "rb") as f:
        data = f.read()

    lines = []
    lines.append("// Auto-generated from har_cnn_int8.tflite - do not edit by hand")
    lines.append("// Regenerate with generate_model_header.py")
    lines.append("#ifndef HAR_MODEL_DATA_H")
    lines.append("#define HAR_MODEL_DATA_H")
    lines.append("")
    lines.append(f"const unsigned int {ARRAY_NAME}_len = {len(data)};")
    lines.append(f"alignas(8) const unsigned char {ARRAY_NAME}[] = {{")

    hex_bytes = [f"0x{b:02x}" for b in data]
    for i in range(0, len(hex_bytes), 12):
        lines.append("  " + ", ".join(hex_bytes[i:i + 12]) + ",")

    lines.append("};")
    lines.append("")
    lines.append("#endif  // HAR_MODEL_DATA_H")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUTPUT_PATH}: {len(data)} bytes -> array '{ARRAY_NAME}'")


if __name__ == "__main__":
    main()