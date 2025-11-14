# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <tuple>
#include <libcamera/libcamera.h>
#include <thread>
#include <chrono>
#include <string>
#include <array>
#include <cmath>


const std::string MODEL = "mobileone_s0_gaze.onnx";
const int SOURCE = 0;  // e.g. camera index for cv::VideoCapture

constexpr double PI = 3.14159265358979323846;

double angular_error(const std::array<double, 3>& gaze_vector,
                     const std::array<double, 3>& label_vector)
{
    // dot product
    double dot_product =
        gaze_vector[0] * label_vector[0] +
        gaze_vector[1] * label_vector[1] +
        gaze_vector[2] * label_vector[2];

    // norms
    double norm_gaze = std::sqrt(
        gaze_vector[0] * gaze_vector[0] +
        gaze_vector[1] * gaze_vector[1] +
        gaze_vector[2] * gaze_vector[2]
    );

    double norm_label = std::sqrt(
        label_vector[0] * label_vector[0] +
        label_vector[1] * label_vector[1] +
        label_vector[2] * label_vector[2]
    );

    double norm_product = norm_gaze * norm_label;

    // avoid division by zero
    if (norm_product == 0.0) {
        return 0.0;  // or handle as you prefer
    }

    double cosine_similarity = dot_product / norm_product;

    // replicate Python: clamp upper bound to 0.9999999
    if (cosine_similarity > 0.9999999)
        cosine_similarity = 0.9999999;

    // (optional & safer) clamp lower bound too:
    if (cosine_similarity < -1.0)
        cosine_similarity = -1.0;

    double angle_rad = std::acos(cosine_similarity);
    double angle_deg = angle_rad * 180.0 / PI;

    return angle_deg;
}

std::array<double, 3> gaze_to_3d(const std::array<double, 2>& gaze)
{
    double yaw   = gaze[0];
    double pitch = gaze[1];

    std::array<double, 3> gaze_vector;
    gaze_vector[0] = -std::cos(pitch) * std::sin(yaw);
    gaze_vector[1] = -std::sin(pitch);
    gaze_vector[2] = -std::cos(pitch) * std::cos(yaw);

    return gaze_vector;
}

// Draw gaze direction vector on the frame
void draw_gaze(cv::Mat& frame,
               const std::array<int, 4>& bbox,
               float pitch,
               float yaw,
               int thickness = 2,
               const cv::Scalar& color = cv::Scalar(0, 0, 255)) // BGR: red
{
    // Unpack bbox
    int x_min = bbox[0];
    int y_min = bbox[1];
    int x_max = bbox[2];
    int y_max = bbox[3];

    // Calculate center of bounding box
    int x_center = (x_min + x_max) / 2;
    int y_center = (y_min + y_max) / 2;

    // Convert grayscale to BGR if needed
    if (frame.channels() == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }

    // Gaze line length = width of the face box
    int length = x_max - x_min;

    // Same math as Python version
    int dx = static_cast<int>(-length * std::sin(pitch) * std::cos(yaw));
    int dy = static_cast<int>(-length * std::sin(yaw));

    // Start and end points
    cv::Point point1(x_center, y_center);
    cv::Point point2(x_center + dx, y_center + dy);

    // Draw center point
    cv::circle(frame, point1, 4, color, -1);

    // Draw arrow for gaze direction
    cv::arrowedLine(
        frame,
        point1,
        point2,
        color,
        thickness,
        cv::LINE_AA,
        0.25
    );
}


// Draws a bounding box with stylized corner lines
void draw_bbox(cv::Mat& image,
               const std::array<int, 4>& bbox,
               const cv::Scalar& color = cv::Scalar(0, 255, 0),
               int thickness = 2,
               float proportion = 0.2f)
{
    // Unpack bbox
    int x_min = bbox[0];
    int y_min = bbox[1];
    int x_max = bbox[2];
    int y_max = bbox[3];

    int width  = x_max - x_min;
    int height = y_max - y_min;

    int corner_length = static_cast<int>(proportion * std::min(width, height));

    // Draw rectangle border (thin frame)
    cv::rectangle(image,
                  cv::Point(x_min, y_min),
                  cv::Point(x_max, y_max),
                  color,
                  1);

    // --- Stylized corners just like Python version ---

    // Top-left
    cv::line(image, cv::Point(x_min, y_min),
                      cv::Point(x_min + corner_length, y_min),
                      color, thickness);
    cv::line(image, cv::Point(x_min, y_min),
                      cv::Point(x_min, y_min + corner_length),
                      color, thickness);

    // Top-right
    cv::line(image, cv::Point(x_max, y_min),
                      cv::Point(x_max - corner_length, y_min),
                      color, thickness);
    cv::line(image, cv::Point(x_max, y_min),
                      cv::Point(x_max, y_min + corner_length),
                      color, thickness);

    // Bottom-left
    cv::line(image, cv::Point(x_min, y_max),
                      cv::Point(x_min, y_max - corner_length),
                      color, thickness);
    cv::line(image, cv::Point(x_min, y_max),
                      cv::Point(x_min + corner_length, y_max),
                      color, thickness);

    // Bottom-right
    cv::line(image, cv::Point(x_max, y_max),
                      cv::Point(x_max, y_max - corner_length),
                      color, thickness);
    cv::line(image, cv::Point(x_max, y_max),
                      cv::Point(x_max - corner_length, y_max),
                      color, thickness);
}


void draw_bbox_gaze(cv::Mat& frame,
                    const std::array<int, 4>& bbox,
                    float pitch,
                    float yaw)
{
    draw_bbox(frame, bbox);
    draw_gaze(frame, bbox, pitch, yaw);
}


#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <utility>   // std::pair

class GazeEstimationONNX {
public:
    // Constructor: loads the ONNX model and prepares everything
    explicit GazeEstimationONNX(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "gaze_estimation"),
          allocator_(),
          session_(nullptr),
          bins_(90),
          binwidth_(4.0f),
          angle_offset_(180.0f),
          input_size_(448, 448),   // width, height
          input_mean_{0.485f, 0.456f, 0.406f},
          input_std_{0.229f, 0.224f, 0.225f}
    {
        // --- Session options (CPU only – good for Raspberry Pi) ---
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED
        );
        // No CUDA provider here -> CPUExecutionProvider only (default)

        session_ = std::make_unique<Ort::Session>(
            env_,
            model_path.c_str(),
            session_options
        );

        // idx_tensor = [0, 1, 2, ..., bins-1]
        idx_tensor_.resize(bins_);
        for (int i = 0; i < bins_; ++i) {
            idx_tensor_[i] = static_cast<float>(i);
        }

        // ---- Input metadata ----
        // If you know the input name is "input", you can simply use that.
        // Here we query the name for generality.
        size_t num_inputs = session_->GetInputCount();
        if (num_inputs != 1) {
            throw std::runtime_error("Expected exactly 1 input for gaze model.");
        }

        auto input_name_alloc = session_->GetInputNameAllocated(0, allocator_);
        input_name_ = input_name_alloc.get();  // copy into std::string

        // You *could* query input shape here; we already know it's 1x3x448x448
        // so we just keep input_size_ = (448, 448).

        // ---- Output metadata ----
        size_t num_outputs = session_->GetOutputCount();
        if (num_outputs != 2) {
            throw std::runtime_error("Expected 2 output nodes (pitch, yaw).");
        }

        output_names_str_.resize(num_outputs);
        output_names_.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto out_name_alloc = session_->GetOutputNameAllocated(i, allocator_);
            output_names_str_[i] = out_name_alloc.get();
            output_names_[i] = output_names_str_[i].c_str();
        }
    }

    // Estimate gaze from a face image (BGR cv::Mat).
    // Returns (pitch, yaw) in radians.
    std::pair<float, float> estimate(const cv::Mat& face_image) {
        // 1) Preprocess -> float tensor [1, 3, H, W]
        std::vector<float> input_tensor_values = preprocess(face_image);

        std::array<int64_t, 4> input_shape = {1, 3,
                                              input_size_.height,
                                              input_size_.width};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 2) Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            &input_name_,
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size()
        );

        if (output_tensors.size() != 2) {
            throw std::runtime_error("Expected two outputs from gaze model.");
        }

        // 3) Extract logits (assuming shape [1, bins_])
        std::vector<float> pitch_logits(bins_);
        std::vector<float> yaw_logits(bins_);

        {
            float* pitch_data = output_tensors[0].GetTensorMutableData<float>();
            float* yaw_data   = output_tensors[1].GetTensorMutableData<float>();
            std::copy(pitch_data, pitch_data + bins_, pitch_logits.begin());
            std::copy(yaw_data,   yaw_data   + bins_, yaw_logits.begin());
        }

        // 4) Decode to (pitch, yaw) in radians
        return decode(pitch_logits, yaw_logits);
    }

private:
    // -------- Members --------
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;

    int bins_;
    float binwidth_;
    float angle_offset_;
    std::vector<float> idx_tensor_;

    cv::Size input_size_;  // (width, height)
    std::array<float, 3> input_mean_;
    std::array<float, 3> input_std_;

    std::string input_name_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;

    // -------- Helper methods --------

    // Preprocess: BGR cv::Mat -> normalized NCHW float tensor [1,3,H,W]
    std::vector<float> preprocess(const cv::Mat& image_bgr) {
        cv::Mat img;
        // Convert BGR -> RGB
        cv::cvtColor(image_bgr, img, cv::COLOR_BGR2RGB);

        // Resize to model input
        cv::resize(img, img, input_size_);

        // Convert to float [0,1]
        img.convertTo(img, CV_32FC3, 1.0f / 255.0f);

        // Normalize per channel
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);  // channels: R, G, B

        for (int c = 0; c < 3; ++c) {
            channels[c] = (channels[c] - input_mean_[c]) / input_std_[c];
        }

        cv::merge(channels, img);  // back to 3-channel float image

        // Convert HWC -> CHW and flatten
        int h = img.rows;
        int w = img.cols;
        std::vector<float> tensor_data(1 * 3 * h * w);

        // img is [H, W, C], we convert to [C, H, W]
        std::vector<cv::Mat> chw(3);
        cv::split(img, chw);  // now each is [H, W] float

        int channel_size = h * w;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(
                tensor_data.data() + c * channel_size,
                chw[c].data,
                channel_size * sizeof(float)
            );
        }

        return tensor_data;
    }

    // Softmax for a single 1D vector of logits
    std::vector<float> softmax(const std::vector<float>& x) const {
        std::vector<float> y(x.size());
        float max_val = *std::max_element(x.begin(), x.end());

        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            y[i] = std::exp(x[i] - max_val);
            sum += y[i];
        }

        if (sum > 0.0f) {
            for (auto& v : y) {
                v /= sum;
            }
        }

        return y;
    }

    // Decode pitch & yaw logits into radians (like Python's decode)
    std::pair<float, float> decode(const std::vector<float>& pitch_logits,
                                   const std::vector<float>& yaw_logits) const
    {
        auto pitch_probs = softmax(pitch_logits);
        auto yaw_probs   = softmax(yaw_logits);

        float pitch_deg = 0.0f;
        float yaw_deg   = 0.0f;

        for (int i = 0; i < bins_; ++i) {
            pitch_deg += pitch_probs[i] * idx_tensor_[i];
            yaw_deg   += yaw_probs[i]   * idx_tensor_[i];
        }

        pitch_deg = pitch_deg * binwidth_ - angle_offset_;
        yaw_deg   = yaw_deg   * binwidth_ - angle_offset_;

        // Convert degrees -> radians
        float pitch_rad = pitch_deg * static_cast<float>(M_PI) / 180.0f;
        float yaw_rad   = yaw_deg   * static_cast<float>(M_PI) / 180.0f;

        return {pitch_rad, yaw_rad};
    }
};

#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <iostream>
#include <array>

// forward declarations (or include your headers)
std::pair<float, float> /* or whatever */ ;
void draw_bbox_gaze(cv::Mat& frame,
                    const std::array<int, 4>& bbox,
                    float pitch,
                    float yaw);

// const std::string MODEL = "mobileone_s0_gaze.onnx";
// const int SOURCE = 0;

int main() {
    try {
        // --- 1. Create gaze engine (loads ONNX model) ---
        GazeEstimationONNX engine(MODEL);

        // --- 2. Load Haar cascade for face detection ---
        // You MUST adjust this path to where your haarcascades actually are.
        // Typical locations:
        //   /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
        //   /usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml
        std::string face_cascade_path =
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

        cv::CascadeClassifier face_cascade;
        if (!face_cascade.load(face_cascade_path)) {
            std::cerr << "Error loading face cascade from: "
                      << face_cascade_path << std::endl;
            return -1;
        }

        // --- 3. Open camera (replaces Picamera2) ---
        cv::VideoCapture cap(SOURCE);  // typically 0
        if (!cap.isOpened()) {
            std::cerr << "Error: could not open camera with index " << SOURCE << std::endl;
            return -1;
        }

        // Set resolution similar to your Python config (640x480)
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        // Warm-up
        std::this_thread::sleep_for(std::chrono::seconds(2));

        cv::Mat frame;

        // --- 4. Capture loop ---
        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "Error: empty frame from camera" << std::endl;
                break;
            }

            // Convert to grayscale for face detection
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(
                gray,
                faces,
                1.1,              // scaleFactor
                5,                // minNeighbors
                0,                // flags
                cv::Size(60, 60)  // minSize
            );

            // For each detected face
            for (const auto& face : faces) {
                int x_min = face.x;
                int y_min = face.y;
                int x_max = face.x + face.width;
                int y_max = face.y + face.height;

                // Safe ROI
                cv::Rect roi(x_min, y_min, face.width, face.height);
                if (roi.x < 0 || roi.y < 0 ||
                    roi.x + roi.width > frame.cols ||
                    roi.y + roi.height > frame.rows) {
                    continue;
                }

                cv::Mat face_crop = frame(roi);
                if (face_crop.empty()) {
                    continue;
                }

                // Estimate gaze (pitch, yaw in radians)
                auto [pitch, yaw] = engine.estimate(face_crop);

                // Draw bbox + gaze on the main frame
                std::array<int, 4> bbox = {x_min, y_min, x_max, y_max};
                draw_bbox_gaze(frame, bbox, pitch, yaw);
            }

            // Show result (needs X/Wayland; if you're on pure console, you'll need another preview method)
            cv::imshow("Gaze Estimation", frame);
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 27) { // 'q' or ESC
                break;
            }
        }

        // --- 5. Cleanup ---
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}



