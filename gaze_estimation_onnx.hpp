// gaze_estimation_onnx.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include <vector>
#include <string>
#include <array>
#include <memory>
#include <utility>   // std::pair

class GazeEstimationONNX {
public:
    explicit GazeEstimationONNX(const std::string& model_path);

    // Estimate gaze from a face image (BGR cv::Mat).
    // Returns (pitch, yaw) in radians.
    std::pair<float, float> estimate(const cv::Mat& face_image);

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
    std::vector<float> preprocess(const cv::Mat& image_bgr);
    std::vector<float> softmax(const std::vector<float>& x) const;
    std::pair<float, float> decode(const std::vector<float>& pitch_logits,
                                   const std::vector<float>& yaw_logits) const;
};
