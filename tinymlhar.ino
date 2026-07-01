#include <Wire.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "har_model_data.h"
#include "har_fixtures.h"

// MPU-6050 I2C config (unused by loop() right now - inference runs off
// fixture_data below. Kept here in case real-sensor input is wired in later.)
#define MPU_ADDR 0x68
#define ACCEL_XOUT_H 0x3B
void mpu_read(float* ax, float* ay, float* az, float* gx, float* gy, float* gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  int16_t raw_ax = Wire.read() << 8 | Wire.read();
  int16_t raw_ay = Wire.read() << 8 | Wire.read();
  int16_t raw_az = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read();  // skip temperature
  int16_t raw_gx = Wire.read() << 8 | Wire.read();
  int16_t raw_gy = Wire.read() << 8 | Wire.read();
  int16_t raw_gz = Wire.read() << 8 | Wire.read();
  *ax = raw_ax / 16384.0f;
  *ay = raw_ay / 16384.0f;
  *az = raw_az / 16384.0f;
  *gx = raw_gx / 131.0f;
  *gy = raw_gy / 131.0f;
  *gz = raw_gz / 131.0f;
}

// Labels
const char* LABELS[] = {
  "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
  "SITTING", "STANDING", "LAYING"
};

// Model config
constexpr int TENSOR_ARENA_SIZE = 12 * 1024;  // measured arena_used_bytes() was 6.93 KB on real hardware sim; sized with headroom, not guessed
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
const tflite::Model* model_ptr = nullptr;

// Ops actually used by har_cnn_int8.tflite, verified directly against the
// converted model (TFLITE_BUILTINS_INT8, zero Flex/Select ops):
// EXPAND_DIMS, CONV_2D, RESHAPE, MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX
tflite::MicroMutableOpResolver<7> resolver;
tflite::MicroInterpreter* interpreter = nullptr;

// Model input shape: 128 timesteps x 9 channels
const int N_STEPS = 128;
const int N_FEATURES = 9;

// Quantization params from har_cnn_int8.tflite (printed by check_int8_accuracy.py).
// These MUST match the model's actual input tensor quantization - if you
// regenerate the model, re-check these values, don't assume they stay the same.
const float INPUT_SCALE = 0.02146185375750065f;
const int INPUT_ZERO_POINT = -15;

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("HAR TinyML CNN — ESP32 (fixture-driven inference)");

  model_ptr = tflite::GetModel(g_har_cnn_model_data);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: model schema version mismatch!");
    return;
  }

  resolver.AddExpandDims();
  resolver.AddConv2D();
  resolver.AddReshape();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interp(
      model_ptr, resolver,
      tensor_arena, (size_t)TENSOR_ARENA_SIZE);
  interpreter = &static_interp;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    return;
  }

  Serial.print("Arena used: ");
  Serial.print(interpreter->arena_used_bytes() / 1024.0);
  Serial.println(" KB");

  TfLiteTensor* input = interpreter->input(0);
  Serial.print("Input tensor type: ");
  Serial.println(input->type == kTfLiteInt8 ? "INT8" : "OTHER (unexpected!)");

  Serial.println("Cycling through 6 activity fixtures...");
}

int current_fixture = 0;

void loop() {
  TfLiteTensor* input = interpreter->input(0);
  int8_t* input_data = input->data.int8;

  // Quantize the float fixture data into the int8 input tensor:
  // q = round(float_value / scale) + zero_point
  for (int t = 0; t < N_STEPS; t++) {
    for (int f = 0; f < N_FEATURES; f++) {
      float val = fixture_data[current_fixture][t][f];
      int32_t q = (int32_t)roundf(val / INPUT_SCALE) + INPUT_ZERO_POINT;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input_data[t * N_FEATURES + f] = (int8_t)q;
    }
  }

  unsigned long t_start = micros();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long t_end = micros();

  if (status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed!");
  } else {
    TfLiteTensor* output = interpreter->output(0);
    int8_t* probs_q = output->data.int8;
    float out_scale = output->params.scale;
    int out_zero_point = output->params.zero_point;

    int predicted = 0;
    float best_prob = -1.0f;
    float probs[6];
    for (int i = 0; i < 6; i++) {
      probs[i] = (probs_q[i] - out_zero_point) * out_scale;
      if (probs[i] > best_prob) {
        best_prob = probs[i];
        predicted = i;
      }
    }

    Serial.print("True: ");
    Serial.print(FIXTURE_LABELS[current_fixture]);
    Serial.print("  |  Predicted: ");
    Serial.print(LABELS[predicted]);
    Serial.print("  (");
    Serial.print(best_prob * 100, 1);
    Serial.print("%)  Latency: ");
    Serial.print((t_end - t_start) / 1000.0, 2);
    Serial.println(" ms");
  }

  current_fixture = (current_fixture + 1) % NUM_FIXTURES;
  delay(3000);
}