#ifndef IMAGERETRIEVER_H
#define IMAGERETRIEVER_H


#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "ImageSet.h"
#include "Vocabulary.h"


class ImageRetriever
{
public:

	struct RetrieverResult {
		RetrieverResult(){

		}
		RetrieverResult(std::string path, double similiarity_score, unsigned matching_points, unsigned cat) :
			path(path), similiarity_score(similiarity_score), matching_points(matching_points), cat(cat)
		{ }
		double similiarity_score;
		unsigned matching_points;
		std::string path;
		unsigned cat;
	};

	ImageRetriever(const cv::Ptr<cv::FeatureDetector>& detector, const cv::Ptr<cv::DescriptorExtractor>& extractor,
		const cv::Ptr<cv::DescriptorMatcher>& matcher, const Vocabulary& vocabulary);


	void train(const std::vector<ImageSet>& image_sets);


	void save() const;

	void load(int category = 0);


	std::vector<RetrieverResult> retrieveSimilarImages(const cv::Mat& image, int category, bool ransac = false);



private:
	struct OrderAux {
		OrderAux(unsigned pos, double val) : pos(pos), val(val)
		{ }
		unsigned pos;
		double val;

	};

	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	cv::Ptr<cv::DescriptorMatcher> matcher;
 
	

	Vocabulary vocabulary;

	int noImgs;
	std::vector < std::string > imagesTopath;
	std::vector <int> imagesToCat;
	std::vector <cv::Mat> bagOfWordsDb;

	std::vector < std::vector <int> > invFileIndex;
	std::vector< double > tfIdfPre;

	std::vector< std::vector<cv::KeyPoint> > keyPointsVector;
	std::vector<cv::Mat> descriptorsVector;

	void saveInvertedFileIndex(cv::FileStorage& fs) const;
	void loadInvertedFileIndex(cv::FileStorage& fs);


	void savekeyPointsVector(cv::FileStorage& fs) const;
	void loadkeyPointsVector(cv::FileStorage& fs);

	void saveKeyPointsAndDescriptors() const;
	void loadKeyPointsAndDescriptors(int imgNo);

	std::vector<int> computeMostRelevantByTfidf(const cv::Mat& desc);

	cv::Mat keyPointsToMat(const std::vector< cv::KeyPoint >& vec) const;
	std::vector<cv::KeyPoint> matToKeypoints(const cv::Mat& mat) const;
};

#endif
