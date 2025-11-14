// main.cpp
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <iostream>
#include <array>

#include "gaze_utils.hpp"
#include "gaze_estimation_onnx.hpp"

const std::string MODEL = "mobileone_s0_gaze.onnx";
const int SOURCE = 0;  // camera index

int main() {
    try {
        GazeEstimationONNX engine(MODEL);

        // Adjust this path for your system
        std::string face_cascade_path ="/opt/homebrew/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";


        cv::CascadeClassifier face_cascade;
        if (!face_cascade.load(face_cascade_path)) {
            std::cerr << "Error loading face cascade from: "
                      << face_cascade_path << std::endl;
            return -1;
        }

        cv::VideoCapture cap(SOURCE);
        if (!cap.isOpened()) {
            std::cerr << "Error: could not open camera with index " << SOURCE << std::endl;
            return -1;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

        std::this_thread::sleep_for(std::chrono::seconds(2));

        cv::Mat frame;

        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "Error: empty frame from camera" << std::endl;
                break;
            }

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(
                gray,
                faces,
                1.1,
                5,
                0,
                cv::Size(60, 60)
            );

            for (const auto& face : faces) {
                int x_min = face.x;
                int y_min = face.y;
                int x_max = face.x + face.width;
                int y_max = face.y + face.height;

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

                auto [pitch, yaw] = engine.estimate(face_crop);

                std::array<int, 4> bbox = {x_min, y_min, x_max, y_max};
                draw_bbox_gaze(frame, bbox, pitch, yaw);
            }

            cv::imshow("Gaze Estimation", frame);
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 27) {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
