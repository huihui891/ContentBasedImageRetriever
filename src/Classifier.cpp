#include "Classifier.h"

#include <iostream>
#include <iomanip>

#include <opencv2/nonfree/nonfree.hpp>

#include "Constants.h"
#include "Helpers.h"
#include "ImageRetriever.h"



using namespace cv;



Classifier::Classifier(const Ptr<FeatureDetector>& detector, const Ptr<DescriptorExtractor>& extractor, const Ptr<DescriptorMatcher>& matcher,
						const Vocabulary& vocabulary)
		: detector(detector), extractor(extractor), matcher(matcher), vocabulary(vocabulary)
{ }


void Classifier::train(const std::vector<ImageSet>& test_sets)
{
	std::cout << "Processing images for classifier..." << std::endl;

	Mat trainingData;
	std::vector<float> trainingLabels;

	for (const auto& test_set : test_sets) {
		for (const auto& image_path : test_set.getImagesPaths()) {

			std::cout << "Processing image " << image_path << std::endl;

			// Open image
			Mat image = openImage(image_path,
				(Constants::CLASSIFIER_USE_COLOR ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE));

			Mat color_hist_descriptor;

			if (Constants::CLASSIFIER_USE_COLOR) {
				// Compute color histograms
				calculateColorHistogramsDescriptor(image, color_hist_descriptor);
				cvtColor(image, image, CV_RGB2GRAY);
			}

			Mat descriptor = vocabulary.computeDescriptor(image);

			if (Constants::CLASSIFIER_USE_COLOR) {
				// Join BoW with color histograms to get a single descriptor for the image
				Mat new_descriptor = joinDescriptors(descriptor, color_hist_descriptor);
				descriptor = new_descriptor;
			}

			trainingData.push_back(descriptor);
			trainingLabels.push_back(static_cast<float>(test_set.getCategory()));
		}
	}

	std::cout << "Training classifier..." << std::endl;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;

	//svm.train(trainingData, cv::Mat(trainingLabels, true), Mat(), Mat(), params);

	svm.train_auto(trainingData, cv::Mat(trainingLabels, true), Mat(), Mat(), params);


	std::cout << "Classifier trained." << std::endl;
}



void Classifier::save(const std::string& filename) const
{
	svm.save(filename.c_str());
}

void Classifier::load(const std::string& filename)
{
	svm.load(filename.c_str());
}


int Classifier::predict(const cv::Mat& image)
{

	Mat descriptor;

	// Compute BoW
	if (Constants::CLASSIFIER_USE_COLOR) {
		Mat gray;
		cvtColor(image, gray, CV_RGB2GRAY);
		descriptor = vocabulary.computeDescriptor(gray);
	} else {
		descriptor = vocabulary.computeDescriptor(image);
	}

	if (Constants::CLASSIFIER_USE_COLOR) {
		// Compute color histograms
		Mat color_hist_descriptor;
		calculateColorHistogramsDescriptor(image, color_hist_descriptor);

		// Join BoW with color histograms to get a single descriptor for the image
		Mat new_descriptor = joinDescriptors(descriptor, color_hist_descriptor);
		descriptor = new_descriptor;
	}

	float prediction = svm.predict(descriptor);

	//std::cout << "Result: " << prediction << std::endl;

	return static_cast<int>(prediction);
}


int Classifier::predictWithRetriever(const cv::Mat& image, ImageRetriever& image_retriever)
{
	std::vector<ImageRetriever::RetrieverResult> results =
		image_retriever.retrieveSimilarImages(image, -1, Constants::RETRIEVE_WITH_RANSAC);

	unsigned category_count[Constants::NUM_CATEGORIES] = { 0 };
	int best_category = 0;
	unsigned* best_category_count = &category_count[0];


	// get the category with the most images
	for (const auto& result : results) {
		category_count[result.cat]++;
		if (category_count[result.cat] > *best_category_count) {
			best_category = result.cat;
			best_category_count = &category_count[result.cat];
		}
	}

	return best_category;
}



void Classifier::evaluate(const std::vector<ImageSet>& eval_sets, ImageRetriever* image_retriever)
{
	std::cout << "Evaluating classifier..." << std::endl;

	Mat trainingData;
	std::vector<float> trainingLabels;

	unsigned total_predictions = 0;
	unsigned correct_predictons = 0;


	unsigned confusion_matrix[Constants::NUM_CATEGORIES][Constants::NUM_CATEGORIES] = { { 0 } };


	for (const auto& eval_set : eval_sets) {
		for (const auto& image_path : eval_set.getImagesPaths()) {

			// Open image
			Mat image;
			if (!openImage(image_path,
				((Constants::CLASSIFIER_USE_COLOR && image_retriever==0) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE),
				image, false)) {
				continue;
			}

			std::cout << "Processing image " << image_path << std::endl;

			int category;
			
			// Predict the category
			if (image_retriever == 0) {
				category = predict(image);
			} else {
				category = predictWithRetriever(image, *image_retriever);
			}


			// Update the confusion matrix

			total_predictions++;

			confusion_matrix[eval_set.getCategory()][category]++;

			if ( category == eval_set.getCategory()) {
				correct_predictons++;
			} else {
				std::cout << "Image " << image_path << " mistaken for " << Constants::CATEGORIES[category] << "." << std::endl;
			}

		}
	}


	std::cout << "Classifier evaluation finished." << std::endl;

	std::cout << "\nConfusion matrix: " << std::endl;

	float accuracy_total = 0.f;

	// Print the confusion matrix and calculate accuracy for each category.

	for (unsigned i = 0; i < Constants::NUM_CATEGORIES; ++i) {
		unsigned total = 0;
		for (unsigned j = 0; j < Constants::NUM_CATEGORIES; ++j) {
			std::cout << confusion_matrix[i][j] << '\t';
			total += confusion_matrix[i][j];
		}
		float accuracy = static_cast<float>(confusion_matrix[i][i]) / total;
		std::cout << "| " << accuracy << std::endl;
		accuracy_total += accuracy;
	}

	float reliability_total = 0.f;

	std::cout << std::endl;

	// Compute and print the reliability for each category

	for (unsigned i = 0; i < Constants::NUM_CATEGORIES; ++i) {
		unsigned total = 0;
		for (unsigned j = 0; j < Constants::NUM_CATEGORIES; ++j) {
			total += confusion_matrix[j][i];
		}
		float reliability = static_cast<float>(confusion_matrix[i][i]) / total;
		std::cout << std::setprecision(4) << reliability << '\t';
		reliability_total += reliability;
	}


	std::cout << "\n\nTotal predictions: " << total_predictions << std::endl
		<< "Correct predictions: " << correct_predictons << std::endl
		<< "\nOverall accuracy: " << static_cast<float>(correct_predictons) / total_predictions << std::endl
		<< "Average accuracy: " << accuracy_total / Constants::NUM_CATEGORIES << std::endl
		<< "Average reliability: " << reliability_total / Constants::NUM_CATEGORIES << std::endl;

}

