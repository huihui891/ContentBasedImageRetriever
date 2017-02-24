#include "Helpers.h"

#include <iostream>
#include <stdexcept>

using namespace cv;


cv::Mat openImage(const std::string& filename, int flags) {
	Mat image = imread(filename, flags);
	if (!image.data) {
		throw std::runtime_error("Error reading image " + filename);
	}
	return image;
}


bool openImage(const std::string& filename, int flags, cv::Mat& image, bool printError)
{
	image = imread(filename, flags);
	if (!image.data) {
		if (printError) {
			std::cout << "Error reading image " << filename << std::endl;
		}
		return false;
	}
	return true;
}



void calculateColorHistograms(const cv::Mat& image, cv::Mat& red_hist, cv::Mat& green_hist, cv::Mat& blue_hist)
{
	/// Separate the image in 3 places ( B, G and R )
	std::vector<Mat> bgr_planes;
	split(image, bgr_planes);

	/// Establish the number of bins
	const int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, histSize };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), blue_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), green_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), red_hist, 1, &histSize, &histRange, uniform, accumulate);


	/// Normalize the result
	normalize(blue_hist, blue_hist);
	normalize(green_hist, green_hist);
	normalize(red_hist, red_hist);
}


void calculateColorHistogramsDescriptor(const cv::Mat& image, cv::Mat& descriptor)
{
	Mat red_hist, green_hist, blue_hist;
	calculateColorHistograms(image, red_hist, green_hist, blue_hist);

	Mat b_hist(1, blue_hist.rows, blue_hist.type());
	Mat g_hist(1, green_hist.rows, green_hist.type());
	Mat r_hist(1, red_hist.rows, red_hist.type());

	for (int i = 0; i < blue_hist.rows; i++) {
		r_hist.at<float>(i) = red_hist.at<float>(i);
		g_hist.at<float>(i) = green_hist.at<float>(i);
		b_hist.at<float>(i) = blue_hist.at<float>(i);
	}

	// join all histograms in a single descriptor
	Mat d1 = joinDescriptors(r_hist, g_hist);
	descriptor = joinDescriptors(d1, b_hist);
	
}



cv::Mat joinDescriptors(const cv::Mat& descriptor1, const cv::Mat& descriptor2)
{
	Mat new_descriptor(1, descriptor1.cols + descriptor2.cols, descriptor1.type());

	descriptor1.copyTo(new_descriptor(Rect(0, 0, descriptor1.cols, 1)));
	descriptor2.copyTo(new_descriptor(Rect(descriptor1.cols, 0, descriptor2.cols, 1)));

	return new_descriptor;
}

