#include <iostream>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "Constants.h"
#include "Helpers.h"
#include "ImageSet.h"
#include "Vocabulary.h"
#include "Classifier.h"
#include "ImageRetriever.h"




using namespace cv;
using namespace std;


void printUsage(const std::string& program_name) {
	std::cout << "Usage:" << std::endl
		<< "\t" << program_name << " mode [arguments]\n\n" << std::endl
		<< "Mode can be:" << std::endl
		<< "\n\tretrieve or -r: classifies an image and retrieves the ones that are most similir to it." << std::endl
		<< "The argument is the path to the image." << std::endl
		<< "\n\ttrain-vocabulary or -v: trains the vocabulary. The first argument is the path to where the images of all the categories are located, "
		<< "the second argument is the first image to use in the training set and the third argument is the number of images to use." << std::endl
		<< "\n\ttrain-classifier or -c: trains the classifier. The first argument is the path to where the images of all the categories are located, "
		<< "the second argument is the first image to use in the training set and the third argument is the number of images to use." << std::endl
		<< "\n\ttrain-database or -d: pre-computes the database. The first argument is the path to where the images of all the categories are located, "
		<< "the second argument is the first image and the third argument is the number of images to use." << std::endl
		<< "\n\tevaluate or -e: evaluates the classifier. The first argument is the path to where the images of all the categories are located, "
		<< "the second argument is the first image to be evaluated and the third argument is the number of images to use." << std::endl;
}


void tryLoad(const std::vector<ImageSet>& test_sets);

std::vector<ImageSet> createImageSets(char** argv, int init_pos);

void retrieveImages(Vocabulary& vocabulary, Classifier& classifier, ImageRetriever& image_retriever, const std::string& filename);



int main(int argc, char** argv)
{
	if (argc < 2) {
		printUsage(argv[0]);
		return 0;
	}

	initModule_nonfree();

	//Ptr<FeatureDetector> detector = new SIFT(2000);
	//Ptr<DescriptorExtractor> extractor = new SIFT(2000);
	Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

	Vocabulary vocabulary(detector, extractor, matcher);


	const std::string mode = argv[1];

	if (mode == "retrieve" || mode == "-r") {

		if (argc != 3) {
			printUsage(argv[0]);
			return 0;
		}

		vocabulary.load(Constants::VOCABULARY_FILE);

		Classifier classifier(detector, extractor, matcher, vocabulary);
		classifier.load(Constants::CLASSIFIER_FILE);

		ImageRetriever imageRetriever(detector, extractor, matcher, vocabulary);
		imageRetriever.load();

		retrieveImages(vocabulary, classifier, imageRetriever, argv[2]);

		waitKey(0);

	} else if (mode == "train-vocabulary" || mode == "-v") {

		if (argc != 5 ) {
			printUsage(argv[0]);
			return 0;
		}

		std::vector<ImageSet> test_sets = createImageSets(argv, 2);

		vocabulary.train(test_sets);
		vocabulary.save(Constants::VOCABULARY_FILE);

	} else if (mode == "train-classifier" || mode == "-c") {

		if (argc != 5) {
			printUsage(argv[0]);
			return 0;
		}

		std::vector<ImageSet> test_sets = createImageSets(argv, 2);

		vocabulary.load(Constants::VOCABULARY_FILE);

		Classifier classifier(detector, extractor, matcher, vocabulary);
		classifier.train(test_sets);

		classifier.save(Constants::CLASSIFIER_FILE);

	} else if (mode == "train-database" || mode == "-d") {

		if (argc != 5) {
			printUsage(argv[0]);
			return 0;
		}

		std::vector<ImageSet> image_sets = createImageSets(argv, 2);

		vocabulary.load(Constants::VOCABULARY_FILE);

		ImageRetriever imageRetriever(detector, extractor, matcher, vocabulary);

		imageRetriever.train(image_sets);

		imageRetriever.save();

	} else if (mode == "tryload") {

		std::vector<ImageSet> test_sets = createImageSets(argv, 2);

		tryLoad(test_sets);

	} else if (mode == "evaluate" || mode == "-e") {

		if (argc != 5) {
			printUsage(argv[0]);
			return 0;
		}

		std::vector<ImageSet> eval_sets = createImageSets(argv, 2);

		vocabulary.load(Constants::VOCABULARY_FILE);

		Classifier classifier(detector, extractor, matcher, vocabulary);

		if (Constants::CLASSIFY_WITH_RETRIEVER) {
			ImageRetriever imageRetriever(detector, extractor, matcher, vocabulary);
			imageRetriever.load();

			classifier.evaluate(eval_sets, &imageRetriever);
		} else {	
			classifier.load(Constants::CLASSIFIER_FILE);

			classifier.evaluate(eval_sets);
		}

	} else {
		printUsage(argv[0]);
	}


	
 
	return 0;
}


std::vector<ImageSet> createImageSets(char** argv, int init_pos)
{
	std::vector<ImageSet> test_sets;

	for (unsigned i = 0; i < Constants::CATEGORIES.size(); i++) {
		test_sets.emplace_back(i, std::string(argv[init_pos]) + '/' + Constants::CATEGORIES[i] + '/',
					std::atoi(argv[init_pos + 1]), std::atoi(argv[init_pos+2]));
	}

	return test_sets;
}


void tryLoad(const std::vector<ImageSet>& test_sets) {
	for (const auto& test_set : test_sets) {
		for (const auto& image_path : test_set.getImagesPaths()) {

			
			//std::cout << "Opening image " << image_path << std::endl;

			try {
				Mat image = openImage(image_path, CV_LOAD_IMAGE_GRAYSCALE);
			}
			catch (std::runtime_error& e) {
				std::cout << e.what() << std::endl;
			}
		}
	}
}


void retrieveImages(Vocabulary& vocabulary, Classifier& classifier, ImageRetriever& image_retriever, const std::string& filename)
{
	Mat image;
	if (!openImage(filename,
		(Constants::CLASSIFIER_USE_COLOR ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE), image)) {
		return;
	}

	int category;
	if ( Constants::CLASSIFY_WITH_RETRIEVER ) {
		category = classifier.predictWithRetriever(image, image_retriever);
	} else {
		category = classifier.predict(image);
	}

	std::cout << "Category: " << Constants::CATEGORIES[category] << std::endl;

	namedWindow("Input image", CV_WINDOW_KEEPRATIO);
	imshow("Input image", image);

	if (Constants::CLASSIFIER_USE_COLOR) {
		cvtColor(image, image, CV_RGB2GRAY);
	}

	std::vector<ImageRetriever::RetrieverResult> results =
			image_retriever.retrieveSimilarImages(image, category, Constants::RETRIEVE_WITH_RANSAC);

	std::cout << "\nMost similar images: " << std::endl;

	for (unsigned i = 0; i < results.size(); ++i) {
		std::cout << i+1 << ". \"" << results[i].path << "\"."
			<< "Value: " << (Constants::RETRIEVE_WITH_RANSAC ? results[i].matching_points : results[i].similiarity_score) << std::endl;

		const std::string window_name = std::to_string(i+1) + results[i].path;

		Mat result_image;
		if (!openImage(results[i].path, CV_LOAD_IMAGE_COLOR, result_image)) {
			return;
		}

		namedWindow(window_name, CV_WINDOW_KEEPRATIO);
		imshow(window_name, result_image);
	}
}