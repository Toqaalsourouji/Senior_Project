// gaze_utils.cpp
#include "gaze_utils.hpp"

#include <cmath>
#include <algorithm>

constexpr double PI_D = 3.14159265358979323846;

double angular_error(const std::array<double, 3>& gaze_vector,
                     const std::array<double, 3>& label_vector)
{
    double dot_product =
        gaze_vector[0] * label_vector[0] +
        gaze_vector[1] * label_vector[1] +
        gaze_vector[2] * label_vector[2];

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

    if (norm_product == 0.0) {
        return 0.0;
    }

    double cosine_similarity = dot_product / norm_product;

    if (cosine_similarity > 0.9999999)
        cosine_similarity = 0.9999999;
    if (cosine_similarity < -1.0)
        cosine_similarity = -1.0;

    double angle_rad = std::acos(cosine_similarity);
    double angle_deg = angle_rad * 180.0 / PI_D;

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

void draw_gaze(cv::Mat& frame,
               const std::array<int, 4>& bbox,
               float pitch,
               float yaw,
               int thickness,
               const cv::Scalar& color)
{
    int x_min = bbox[0];
    int y_min = bbox[1];
    int x_max = bbox[2];
    int y_max = bbox[3];

    int x_center = (x_min + x_max) / 2;
    int y_center = (y_min + y_max) / 2;

    if (frame.channels() == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }

    int length = x_max - x_min;

    int dx = static_cast<int>(-length * std::sin(pitch) * std::cos(yaw));
    int dy = static_cast<int>(-length * std::sin(yaw));

    cv::Point point1(x_center, y_center);
    cv::Point point2(x_center + dx, y_center + dy);

    cv::circle(frame, point1, 4, color, -1);

    cv::arrowedLine(
        frame,
        point1,
        point2,
        color,
        thickness,
        cv::LINE_AA,
        0,
        0.25
    );
}

void draw_bbox(cv::Mat& image,
               const std::array<int, 4>& bbox,
               const cv::Scalar& color,
               int thickness,
               float proportion)
{
    int x_min = bbox[0];
    int y_min = bbox[1];
    int x_max = bbox[2];
    int y_max = bbox[3];

    int width  = x_max - x_min;
    int height = y_max - y_min;

    int corner_length = static_cast<int>(proportion * std::min(width, height));

    cv::rectangle(image,
                  cv::Point(x_min, y_min),
                  cv::Point(x_max, y_max),
                  color,
                  1);

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
