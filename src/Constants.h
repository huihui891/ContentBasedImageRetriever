#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>


namespace Constants {

	const std::string VOCABULARY_FILE{"vocabulary.xml"};
	const std::string CLASSIFIER_FILE{"classifier.xml"};

	//const std::string FEATURE_DETECTOR_TYPE{"SURF"};
	//const std::string DESCRIPTOR_EXTRACTOR_TYPE{"SURF"};
	//const std::string DESCRIPTOR_MATCHER_TYPE{"FlannBased"};

	const int VOCABULARY_NUM_WORDS = 1000;
	const int VOCABULARY_TRAINER_ATTEMPTS = 1;

	const bool CLASSIFIER_USE_COLOR = true;

	const bool CLASSIFY_WITH_RETRIEVER = false;

	const bool RETRIEVE_WITH_RANSAC = true;
	const unsigned SIMILIAR_IMAGES_RETRIEVE = 10;
	const unsigned IMAGES_TO_RANSAC = 20;
	const unsigned TFIDF_WORDS_CONSIDER = 3;


	const int CATEGORY_CASINO = 0;
	const int CATEGORY_ARCHIVE = 1;
	const int CATEGORY_PARKING = 2;
	const int CATEGORY_RAINFOREST = 3;
	const int CATEGORY_SKI = 4;

	const unsigned NUM_CATEGORIES = 5;

	const std::vector<std::string> CATEGORIES = { "casino", "archive", "parking", "rainforest", "ski" };
	

}


#endif