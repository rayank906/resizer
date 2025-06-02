#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <omp.h>

// energy map
cv::Mat computeEnergyMap(const cv::Mat& gray) {
    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, energy;
    cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::add(abs_grad_x, abs_grad_y, energy);
    return energy;
}

// vertical seam finder function
std::vector<int> findVerticalSeamGreedy(const cv::Mat& energy) {
    int rows = energy.rows;
    int cols = energy.cols;
    std::vector<int> seam(rows);

    int min_col = 0;
    uchar min_val = energy.at<uchar>(0, 0);
    for (int j = 1; j < cols; ++j) {
        if (energy.at<uchar>(0, j) < min_val) {
            min_val = energy.at<uchar>(0, j);
            min_col = j;
        }
    }
    seam[0] = min_col;

    for (int i = 1; i < rows; ++i) {
        int prev_col = seam[i - 1];
        int best_col = prev_col;
        uchar best_energy = energy.at<uchar>(i, prev_col);

        if (prev_col > 0 && energy.at<uchar>(i, prev_col - 1) < best_energy) {
            best_col = prev_col - 1;
            best_energy = energy.at<uchar>(i, prev_col - 1);
        }
        if (prev_col < cols - 1 && energy.at<uchar>(i, prev_col + 1) < best_energy) {
            best_col = prev_col + 1;
        }
        seam[i] = best_col;
    }
    return seam;
}

// removing vertical seam
void removeVerticalSeam(cv::Mat& image, const std::vector<int>& seam) {
    int rows = image.rows;
    int cols = image.cols;

    cv::Mat carved(rows, cols - 1, CV_8UC3);

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        int col = seam[i];
        memcpy(carved.ptr(i), image.ptr(i), col * 3);
        memcpy(carved.ptr(i) + col * 3, image.ptr(i) + (col + 1) * 3, (cols - col - 1) * 3);
    }

    image = carved;
}

// resizing frame
void seamCarveFrame(cv::Mat& frame, int targetWidth, int targetHeight) {
    while (frame.cols > targetWidth) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat energy = computeEnergyMap(gray);
        std::vector<int> seam = findVerticalSeamGreedy(energy);
        removeVerticalSeam(frame, seam);
    }

    frame = frame.t();
    while (frame.cols > targetHeight) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat energy = computeEnergyMap(gray);
        std::vector<int> seam = findVerticalSeamGreedy(energy);
        removeVerticalSeam(frame, seam);
    }
    frame = frame.t();
}

int main() {
    std::string inputPath = "test.mp4";
    std::string outputPath = "output.mp4";
    int targetWidth = 1300;
    int targetHeight = 450;

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << inputPath << std::endl;
        return -1;
    }

    int origWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int origHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(targetWidth, targetHeight)
    );

    std::vector<cv::Mat> frames;

    // Load all frames into memory
    cv::Mat frame;
    while (cap.read(frame)) {
        frames.push_back(frame.clone());
    }

    // Parallel frame-level seam carving
    #pragma omp parallel for
    for (int i = 0; i < frames.size(); ++i) {
        seamCarveFrame(frames[i], targetWidth, targetHeight);
    }

    for (const auto& f : frames) {
        writer.write(f);
    }

    std::cout << "Resized to " << outputPath << std::endl;
    return 0;
}