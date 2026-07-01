"""
Generate har_fixtures.h: one real 128-timestep sample per class, pulled from
X_test.npy / y_test.npy. Matches the existing file's naming convention
(NUM_FIXTURES, FIXTURE_LABELS, fixture_data) so tinymlhar.ino doesn't need
to change.

Run in your repo folder:
    cd /mnt/DATA/SPLAB/tinymlhar/
    python3 generate_fixtures.py
"""

import numpy as np

X_TEST_PATH = "X_test.npy"
Y_TEST_PATH = "y_test.npy"
OUTPUT_PATH = "har_fixtures.h"

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]


def main():
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    # Pick the first test sample for each class (0-5)
    chosen_indices = []
    for c in range(6):
        idx = np.where(y_test == c)[0]
        if len(idx) == 0:
            raise ValueError(f"No test samples found for class {c} ({LABELS[c]})")
        chosen_indices.append(idx[0])

    n_steps = X_test.shape[1]
    n_features = X_test.shape[2]
    print(f"Using one real sample per class, shape ({n_steps}, {n_features})")
    for c, idx in enumerate(chosen_indices):
        print(f"  {LABELS[c]:20s} <- X_test[{idx}]")

    lines = []
    lines.append("#ifndef HAR_FIXTURES_H")
    lines.append("#define HAR_FIXTURES_H")
    lines.append("")
    lines.append(f"const int NUM_FIXTURES = 6;")
    lines.append('const char* FIXTURE_LABELS[6] = {"' + '","'.join(LABELS) + '"};')
    lines.append("")
    lines.append(f"const float fixture_data[6][{n_steps}][{n_features}] = {{")

    for c, idx in enumerate(chosen_indices):
        lines.append(f"  // {LABELS[c]} (real UCI HAR test sample, index {idx})")
        lines.append("  {")
        sample = X_test[idx]
        for t in range(n_steps):
            vals = ", ".join(f"{v:.4f}f" for v in sample[t])
            lines.append(f"    {{{vals}}},")
        lines.append("  },")

    lines.append("};")
    lines.append("")
    lines.append("#endif  // HAR_FIXTURES_H")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()