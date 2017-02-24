#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ImageSet.h"


class Vocabulary
{
public:

	Vocabulary(const cv::Ptr<cv::FeatureDetector>& detector, const cv::Ptr<cv::DescriptorExtractor>& extractor, const cv::Ptr<cv::DescriptorMatcher>& matcher);


	void train(const std::vector<ImageSet>& test_sets);


	bool save(const std::string& filename) const;

	bool load(const std::string& filename);



	cv::Mat computeDescriptor(const cv::Mat& image);

	cv::Mat computeDescriptor(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);


private:

	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	cv::Ptr<cv::DescriptorMatcher> matcher;
	cv::BOWImgDescriptorExtractor bowExtractor;

};

#endif
