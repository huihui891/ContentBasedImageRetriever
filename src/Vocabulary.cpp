#include "Vocabulary.h"

#include <iostream>

#include <opencv2/nonfree/nonfree.hpp>

#include "Constants.h"
#include "Helpers.h"


using namespace cv;



Vocabulary::Vocabulary(const Ptr<FeatureDetector>& detector, const Ptr<DescriptorExtractor>& extractor, const Ptr<DescriptorMatcher>& matcher)
	: detector(detector), extractor(extractor), matcher(matcher), bowExtractor(extractor, matcher)
{ }



void Vocabulary::train(const std::vector<ImageSet>& test_sets)
{
	std::cout << "Training vocabulary..." << std::endl;

	BOWKMeansTrainer bowTrainer(Constants::VOCABULARY_NUM_WORDS, TermCriteria(),
						Constants::VOCABULARY_TRAINER_ATTEMPTS, KMEANS_PP_CENTERS);


	for (const auto& test_set : test_sets) {
		for (const auto& image_path : test_set.getImagesPaths()) {

			std::cout << "Processing image " << image_path << std::endl;

			// Open image
			Mat image = openImage(image_path, CV_LOAD_IMAGE_GRAYSCALE);

			// Detect keypoints
			std::vector<KeyPoint> keypoints;
			detector->detect(image, keypoints);

			// Compute descriptors
			Mat descriptors;
			extractor->compute(image, keypoints, descriptors);

			bowTrainer.add(descriptors);
		}
	}

	std::cout << "Clustering... " << std::endl;

	Mat vocabulary = bowTrainer.cluster();

	bowExtractor.setVocabulary(vocabulary);

	std::cout << "Vocabulary trained." << std::endl;
}



bool Vocabulary::save(const std::string& filename) const
{
	FileStorage fs(filename, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "vocabulary" << bowExtractor.getVocabulary();
		return true;
	}
	return false;
}

bool Vocabulary::load(const std::string& filename)
{
	FileStorage fs(filename, FileStorage::READ);
	if (fs.isOpened())
	{
		Mat vocabulary;
		fs["vocabulary"] >> vocabulary;
		bowExtractor.setVocabulary(vocabulary);
		return true;
	}
	return false;
}



cv::Mat Vocabulary::computeDescriptor(const cv::Mat& image)
{
	// Detect keypoints
	std::vector<KeyPoint> keypoints;
	detector->detect(image, keypoints);

	return computeDescriptor(image, keypoints);
}


cv::Mat Vocabulary::computeDescriptor(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints)
{

	// Compute the bag of words and normalize it
	Mat descriptor;
	bowExtractor.compute(image, keypoints, descriptor);
	normalize(descriptor, descriptor);

	return descriptor;
}
