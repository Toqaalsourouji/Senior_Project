#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_camera.h"
#include "model_data.h"
#include "WiFi.h"

// Camera Pins
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM    5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// TensorFlow Setup
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int8_t *input_data = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroMutableOpResolver<8> resolver;

// Performance Tracking
uint32_t frame_count = 0;
uint32_t last_fps_time = 0;
float current_fps = 0;

void setup() {
  Serial.begin(115200);
  while(!Serial);

  camera_config_t config;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.xclk_freq_hz = 20000000; 
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_96X96;
  config.jpeg_quality = 0;
  config.fb_count = 3; 
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  
  WiFi.mode(WIFI_OFF);
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    return;
  }

  // Allocate tensor arena in PSRAM
  tensor_arena = (uint8_t*) ps_malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena");
    return;
  }

  // Load model
  model = tflite::GetModel(TinyTrackerS_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch");
    return;
  }

  // Setup operations resolver
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddConcatenation();
  resolver.AddTanh();
  resolver.AddDequantize();

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  input_data = (int8_t*) ps_malloc(96 * 96 * sizeof(int8_t));
  
  Serial.println("System initialized. Starting gaze tracking...");
}


void update_peak_memory() {

    Serial.print("DRAM: ");
    Serial.print(ESP.getHeapSize() - ESP.getFreeHeap());
    Serial.print("/");
    Serial.print(ESP.getHeapSize());
    Serial.print(" | PSRAM: ");
    Serial.print(ESP.getPsramSize() - ESP.getFreePsram());
    Serial.print("/");
    Serial.print(ESP.getPsramSize());
    Serial.println();

}

void quantize_image(uint8_t *src, int8_t *dst) {
  const float factor = 1.0f / (255.0f * input->params.scale);
  const int zero_point = input->params.zero_point;
  const int total_pixels = 96 * 96;
  
  for (int i = 0; i < total_pixels; i++) {
    dst[i] = (int8_t)((src[i] - 128) * factor) + zero_point;
  }
}

void loop() {
  uint32_t Loop_start = millis();
  uint32_t capture_start = micros();
  camera_fb_t *fb = esp_camera_fb_get();
  uint32_t capture_time = micros() - capture_start;
  if (!fb) {
    Serial.println("Frame capture failed");
    return;
  }

   uint32_t quant_start = millis();
   quantize_image(fb->buf, input_data);
   uint32_t quant_time = millis() - quant_start;
  
  for(int i=0; i<96*96; i++) {
    input->data.int8[i] = input_data[i];
  }

  uint32_t inference_start = millis();
  interpreter->Invoke();
  uint32_t inference_time = millis() - inference_start;

  // Get results
  float gaze_x = output->data.f[0];
  float gaze_y = output->data.f[1];

  // Release frame buffer immediately
  esp_camera_fb_return(fb);

  // Calculate FPS
  frame_count++;
  uint32_t current_time = millis();
  if(current_time - last_fps_time >= 1000) {
    current_fps = frame_count * 1000.0 / (current_time - last_fps_time);
    frame_count = 0;
    last_fps_time = current_time;
    
    update_peak_memory();
    uint32_t loop_end = millis() - Loop_start;
    Serial.printf("FPS: %.1f | Capture: %luµs | Quant: %lums | Inference: %lums | Loop: %lums\n",
                 current_fps, capture_time, quant_time, inference_time, loop_end);
    Serial.printf("Gaze: (%.2f, %.2f)\n", gaze_x, gaze_y);

  }

}