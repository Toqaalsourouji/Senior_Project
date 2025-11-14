// gaze_estimation_onnx.cpp
#include "gaze_estimation_onnx.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

GazeEstimationONNX::GazeEstimationONNX(const std::string& model_path)
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
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    session_ = std::make_unique<Ort::Session>(
        env_,
        model_path.c_str(),
        session_options
    );

    idx_tensor_.resize(bins_);
    for (int i = 0; i < bins_; ++i) {
        idx_tensor_[i] = static_cast<float>(i);
    }

    size_t num_inputs = session_->GetInputCount();
    if (num_inputs != 1) {
        throw std::runtime_error("Expected exactly 1 input for gaze model.");
    }

    auto input_name_alloc = session_->GetInputNameAllocated(0, allocator_);
    input_name_ = input_name_alloc.get();

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

std::vector<float> GazeEstimationONNX::preprocess(const cv::Mat& image_bgr)
{
    cv::Mat img;
    cv::cvtColor(image_bgr, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, input_size_);
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);  // R, G, B

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - input_mean_[c]) / input_std_[c];
    }

    cv::merge(channels, img);

    int h = img.rows;
    int w = img.cols;
    std::vector<float> tensor_data(1 * 3 * h * w);

    std::vector<cv::Mat> chw(3);
    cv::split(img, chw);

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

std::vector<float> GazeEstimationONNX::softmax(const std::vector<float>& x) const
{
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

std::pair<float, float> GazeEstimationONNX::decode(const std::vector<float>& pitch_logits,
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

    constexpr float DEG2RAD = 3.14159265358979323846f / 180.0f;
    float pitch_rad = pitch_deg * DEG2RAD;
    float yaw_rad   = yaw_deg   * DEG2RAD;

    return {pitch_rad, yaw_rad};
}

std::pair<float, float> GazeEstimationONNX::estimate(const cv::Mat& face_image)
{
    std::vector<float> input_tensor_values = preprocess(face_image);

    std::array<int64_t, 4> input_shape = {
        1,
        3,
        input_size_.height,
        input_size_.width
    };

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

    const char* input_names[] = { input_name_.c_str() };

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size()
    );

    if (output_tensors.size() != 2) {
        throw std::runtime_error("Expected two outputs from gaze model.");
    }

    std::vector<float> pitch_logits(bins_);
    std::vector<float> yaw_logits(bins_);

    float* pitch_data = output_tensors[0].GetTensorMutableData<float>();
    float* yaw_data   = output_tensors[1].GetTensorMutableData<float>();

    std::copy(pitch_data, pitch_data + bins_, pitch_logits.begin());
    std::copy(yaw_data,   yaw_data   + bins_, yaw_logits.begin());

    return decode(pitch_logits, yaw_logits);
}
