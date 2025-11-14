// gaze_utils.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <array>

// Angular error between two 3D gaze vectors (in degrees)
double angular_error(const std::array<double, 3>& gaze_vector,
                     const std::array<double, 3>& label_vector);

// Convert (yaw, pitch) in radians to 3D gaze vector
std::array<double, 3> gaze_to_3d(const std::array<double, 2>& gaze);

// Draw gaze direction vector on the frame
void draw_gaze(cv::Mat& frame,
               const std::array<int, 4>& bbox,
               float pitch,
               float yaw,
               int thickness = 2,
               const cv::Scalar& color = cv::Scalar(0, 0, 255)); // BGR: red

// Draws a bounding box with stylized corner lines
void draw_bbox(cv::Mat& image,
               const std::array<int, 4>& bbox,
               const cv::Scalar& color = cv::Scalar(0, 255, 0),
               int thickness = 2,
               float proportion = 0.2f);

// Convenience: draw both bbox and gaze
void draw_bbox_gaze(cv::Mat& frame,
                    const std::array<int, 4>& bbox,
                    float pitch,
                    float yaw);

                    