#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "ImageSet.h"
#include "Vocabulary.h"


class ImageRetriever;

class Classifier
{
public:

	Classifier(const cv::Ptr<cv::FeatureDetector>& detector, const cv::Ptr<cv::DescriptorExtractor>& extractor, const cv::Ptr<cv::DescriptorMatcher>& matcher,
				const Vocabulary& vocabulary);


	void train(const std::vector<ImageSet>& test_sets);


	void save(const std::string& filename) const;

	void load(const std::string& filename);


	int predict(const cv::Mat& image);

	int predictWithRetriever(const cv::Mat& image, ImageRetriever& image_retriever);


	void evaluate(const std::vector<ImageSet>& eval_sets, ImageRetriever* image_retriever = 0);


private:


	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	cv::Ptr<cv::DescriptorMatcher> matcher;

	Vocabulary vocabulary;
	CvSVM svm;

};

#endif
