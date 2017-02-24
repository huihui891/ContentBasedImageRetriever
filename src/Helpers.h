#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <opencv2/opencv.hpp>


cv::Mat openImage(const std::string& filename, int flags);

bool openImage(const std::string& filename, int flags, cv::Mat& image, bool printError = true);

void calculateColorHistograms(const cv::Mat& image, cv::Mat& red_hist, cv::Mat& green_hist, cv::Mat& blue_hist);

void calculateColorHistogramsDescriptor(const cv::Mat& image, cv::Mat& descriptor);

cv::Mat joinDescriptors(const cv::Mat& descriptor1, const cv::Mat& descriptor2);



#endif