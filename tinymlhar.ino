#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "har_model_data.h" // your C header from step 2
//  Labels 
const char* LABELS[] = {
  "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
  "SITTING", "STANDING", "LAYING"
};

//  Model config 
constexpr int TENSOR_ARENA_SIZE = 60 * 1024;
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model_ptr = nullptr;
tflite::MicroMutableOpResolver<10> resolver;
tflite::MicroInterpreter* interpreter = nullptr;

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println("BOOT OK");
  Serial.flush();
}

void loop() {
  //  Copy test window into input tensor 
  TfLiteTensor* input = interpreter->input(0);
  float* input_data = input->data.f;

  // Hardcoded sample: one 128-step window of WALKING
  // (replace these with real MPU-6050 readings later)
  for (int t = 0; t < 128; t++) {
    input_data[t * 9 + 0] =  0.0288f;  // body_acc_x
    input_data[t * 9 + 1] = -0.0032f;  // body_acc_y
    input_data[t * 9 + 2] = -0.0095f;  // body_acc_z
    input_data[t * 9 + 3] =  0.0050f;  // body_gyro_x
    input_data[t * 9 + 4] = -0.0039f;  // body_gyro_y
    input_data[t * 9 + 5] = -0.0137f;  // body_gyro_z
    input_data[t * 9 + 6] =  1.0128f;  // total_acc_x
    input_data[t * 9 + 7] = -0.0023f;  // total_acc_y
    input_data[t * 9 + 8] =  0.0091f;  // total_acc_z
  }

  //  Run inference 
  unsigned long t_start = micros();
  interpreter->Invoke();
  unsigned long t_end = micros();

  //  Read output
  TfLiteTensor* output = interpreter->output(0);
  float* probs = output->data.f;

  int predicted = 0;
  for (int i = 1; i < 6; i++) {
    if (probs[i] > probs[predicted]) predicted = i;
  }

  // Print result 
  Serial.print("Predicted: ");
  Serial.print(LABELS[predicted]);
  Serial.print("  (");
  Serial.print(probs[predicted] * 100, 1);
  Serial.print("%)  Latency: ");
  Serial.print((t_end - t_start) / 1000.0, 2);
  Serial.println(" ms");

  delay(2000);
}