#include "ImageRetriever.h"

#include <iostream>

#include <opencv2/nonfree/nonfree.hpp>

#include "Helpers.h"
#include "Constants.h"


using namespace cv;
using namespace std;

ImageRetriever::ImageRetriever(const cv::Ptr<cv::FeatureDetector>& detector, const cv::Ptr<cv::DescriptorExtractor>& extractor,
	const cv::Ptr<cv::DescriptorMatcher>& matcher, const Vocabulary& vocabulary)
	: detector(detector), extractor(extractor), matcher(matcher), vocabulary(vocabulary), invFileIndex(Constants::VOCABULARY_NUM_WORDS, vector<int>())
{ }


void ImageRetriever::train(const std::vector<ImageSet>& image_sets)
{
	std::cout << "Creating database..." << std::endl;

	noImgs = 0;
	for (const auto& eval_set : image_sets) {
		vector <Mat> catDb;

		for (const auto& image_path : eval_set.getImagesPaths()) {
			Mat image;

			if (!openImage(image_path, CV_LOAD_IMAGE_GRAYSCALE, image, false)) {
				continue;
			}



			std::vector<KeyPoint> keypoints;
			detector->detect(image, keypoints);

			Mat descriptors;
			extractor->compute(image, keypoints, descriptors);

			keyPointsVector.push_back(keypoints);
			descriptorsVector.push_back(descriptors);

			Mat desc = vocabulary.computeDescriptor(image, keypoints);
			for (size_t i = 0; i < Constants::VOCABULARY_NUM_WORDS; i++)
			{
				if (desc.at<float>(0, i) > 0){
					invFileIndex[i].push_back(noImgs);
				}
			}

			imagesTopath.push_back(image_path);
			imagesToCat.push_back(eval_set.getCategory());
			bagOfWordsDb.push_back(desc);
			std::cout << "Processing image " << image_path << std::endl;

			++noImgs;

		}
	}
	for (auto it = invFileIndex.begin(); it != invFileIndex.end(); ++it){
		sort(it->begin(), it->end());
	}
	tfIdfPre.resize(Constants::VOCABULARY_NUM_WORDS);
	for (size_t i = 0; i < Constants::VOCABULARY_NUM_WORDS; i++)
	{
		tfIdfPre[i] = log((double)noImgs / (double)invFileIndex[i].size());
	}

}

void ImageRetriever::saveInvertedFileIndex(cv::FileStorage& fs) const
{
	fs << "InvertedFileIndex" << "[";
	for (unsigned i = 0; i < invFileIndex.size(); i++)
	{

		fs << invFileIndex[i];
	}
	fs << "]";
}

void ImageRetriever::loadInvertedFileIndex(cv::FileStorage& fs){
	FileNode fnode = fs["InvertedFileIndex"];
	FileNodeIterator it = fnode.begin(), it_end = fnode.end();
	unsigned i = 0;
	for (; it != it_end; ++it, ++i)
	{
		(*it) >> invFileIndex[i];
	}
}



vector<KeyPoint> ImageRetriever::matToKeypoints(const Mat& mat) const {

	vector<KeyPoint>  c_keypoints;

	for (int i = 0; i < mat.rows; i++) {
		Vec<float, 7> v = mat.at< Vec<float, 7> >(i, 0);

		KeyPoint kp(v[0], v[1], v[2], v[3], v[4], (int)v[5], (int)v[6]);

		c_keypoints.push_back(kp);

	};

	return c_keypoints;

};

cv::Mat ImageRetriever::keyPointsToMat(const std::vector<cv::KeyPoint>& vec) const{
	std::vector<cv::Point2f> points;
	for (auto it = vec.begin(); it != vec.end(); it++)
	{
		points.push_back(it->pt);
	}
	cv::Mat pointmatrix(points);
	return pointmatrix.clone();
}
void ImageRetriever::saveKeyPointsAndDescriptors() const
{
	FileStorage fsFor;
	for (unsigned int i = 0; i < keyPointsVector.size(); ++i){
		ostringstream convert;
		convert << i;
		fsFor.open("keyDes/" + convert.str()+".xml.gz", FileStorage::WRITE);
		fsFor << "KeyPoints" << keyPointsToMat(keyPointsVector[i]);
		fsFor << "descriptors" << descriptorsVector[i];
		fsFor.release();
	}
}

void ImageRetriever::loadKeyPointsAndDescriptors(int imgNo){
	FileStorage fsFor;
	ostringstream convert;
	convert << imgNo;
	fsFor.open("keyDes/" + convert.str() + ".xml.gz", FileStorage::READ);
	Mat tmp;
	fsFor["KeyPoints"] >> tmp;
	keyPointsVector[imgNo] = matToKeypoints(tmp);
	fsFor["descriptors"] >> descriptorsVector[imgNo];
	fsFor.release();
}

void ImageRetriever::save() const
{
	FileStorage fs("retrieverDb.xml", FileStorage::WRITE);
	saveInvertedFileIndex(fs);
	fs << "imagesTopath" << imagesTopath;
	fs << "imagesToCat" << imagesToCat;
	fs << "bagOfWordsDb" << bagOfWordsDb;
	fs << "tfIdfPre" << tfIdfPre;
	fs.release();

	saveKeyPointsAndDescriptors();
}


void ImageRetriever::load(int cat)
{
	FileStorage fs("retrieverDb.xml", FileStorage::READ);
	loadInvertedFileIndex(fs);
	fs["imagesTopath"] >> imagesTopath;
	fs["imagesToCat"] >> imagesToCat;
	fs["bagOfWordsDb"] >> bagOfWordsDb;
	fs["tfIdfPre"] >> tfIdfPre;
	fs.release();

	keyPointsVector.resize(imagesTopath.size());
	descriptorsVector.resize(imagesTopath.size());


	//cout << "Loaded Complete\n";
}

Mat filterMatchesRANSAC(std::vector<DMatch> &matches, std::vector<KeyPoint> &keypointsA, std::vector<KeyPoint> &keypointsB)
{
	Mat homography;
	std::vector<DMatch> filteredMatches;
	if (matches.size() >= 4)
	{
		vector<Point2f> srcPoints;
		vector<Point2f> dstPoints;
		for (size_t i = 0; i < matches.size(); i++)
		{

			srcPoints.push_back(keypointsA[matches[i].queryIdx].pt);
			dstPoints.push_back(keypointsB[matches[i].trainIdx].pt);
		}

		Mat mask;
		homography = findHomography(srcPoints, dstPoints, CV_RANSAC, 1.0, mask);

		for (int i = 0; i < mask.rows; i++)
		{
			if (mask.ptr<uchar>(i)[0] == 1)
				filteredMatches.push_back(matches[i]);
		}
	}
	matches = filteredMatches;
	return homography;
}

void showResult(Mat &imgA, std::vector<KeyPoint> &keypointsA, Mat &imgB, std::vector<KeyPoint> &keypointsB,
	std::vector<DMatch> &matches, Mat &homography,
	const std::string& name)
{
	// Draw matches
	Mat imgMatch;
	drawMatches(imgA, keypointsA, imgB, keypointsB, matches, imgMatch);

	if (!homography.empty())
	{
		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(imgA.cols, 0);
		obj_corners[2] = cvPoint(imgA.cols, imgA.rows); obj_corners[3] = cvPoint(0, imgA.rows);
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, homography);

		float cols = (float)imgA.cols;
		line(imgMatch, scene_corners[0] + Point2f(cols, 0), scene_corners[1] + Point2f(cols, 0), Scalar(0, 255, 0), 4);
		line(imgMatch, scene_corners[1] + Point2f(cols, 0), scene_corners[2] + Point2f(cols, 0), Scalar(0, 255, 0), 4);
		line(imgMatch, scene_corners[2] + Point2f(cols, 0), scene_corners[3] + Point2f(cols, 0), Scalar(0, 255, 0), 4);
		line(imgMatch, scene_corners[3] + Point2f(cols, 0), scene_corners[0] + Point2f(cols, 0), Scalar(0, 255, 0), 4);
	}


	namedWindow(name, CV_WINDOW_KEEPRATIO);
	imshow(name, imgMatch);
}


std::vector<ImageRetriever::RetrieverResult> ImageRetriever::retrieveSimilarImages(const cv::Mat& image, int category, bool ransac)
{
	std::vector<KeyPoint> initial_keypoints;
	detector->detect(image, initial_keypoints);


	Mat desc = vocabulary.computeDescriptor(image, initial_keypoints);
	vector<int> imgsNo = computeMostRelevantByTfidf(desc);
	vector<OrderAux> aux;
	for (auto it = imgsNo.begin(); it != imgsNo.end(); ++it){
		if (category < 0 || imagesToCat[*it] == category)
			aux.push_back(OrderAux(*it, bagOfWordsDb[*it].dot(desc)));
	}
	sort(aux.begin(), aux.end(), [](OrderAux order1, OrderAux order2){
		return order1.val > order2.val;
	});

	vector<RetrieverResult> ret;
	if (ransac){
		Mat initial_descriptors;
		extractor->compute(image, initial_keypoints, initial_descriptors);
		unsigned end = min((unsigned)aux.size(), Constants::IMAGES_TO_RANSAC);

		ret.resize(end);
		for (unsigned i = 0; i < end; ++i){
			int imgNo = aux[i].pos;
			loadKeyPointsAndDescriptors(imgNo);

			std::vector< DMatch > matches;
			matcher->match(initial_descriptors, descriptorsVector[imgNo], matches);

			Mat homography;


			homography = filterMatchesRANSAC(matches, initial_keypoints, keyPointsVector[imgNo]);
			ret[i] = RetrieverResult(imagesTopath[imgNo], aux[i].val, matches.size(), imagesToCat[imgNo]);
		}
		sort(ret.begin(), ret.end(), [](RetrieverResult order1, RetrieverResult order2){
			return order1.matching_points > order2.matching_points;
		});
	}
	else
	{
		unsigned end = min((unsigned)aux.size(), Constants::SIMILIAR_IMAGES_RETRIEVE);
		for(unsigned i = 0; i < end; ++i){
			int imgNo = aux[i].pos;
			ret.push_back(RetrieverResult(imagesTopath[imgNo], aux[i].val, 0, imagesToCat[imgNo]));
		}
	}
	//showResult(image.clone(), initial_keypoints, img, keyPointTmp, matches, homography, "Matches ");


	std::vector<RetrieverResult> slice(ret.begin(), ret.begin() + min(Constants::SIMILIAR_IMAGES_RETRIEVE, (unsigned)ret.size()));
	return slice;
}

void inplace_union(std::vector<int>& a, const std::vector<int>& b)
{
	int mid = a.size(); //Store the end of first sorted range

	//First copy the second sorted range into the destination vector
	std::copy(b.begin(), b.end(), std::back_inserter(a));

	//Then perform the in place merge on the two sub-sorted ranges.
	std::inplace_merge(a.begin(), a.begin() + mid, a.end());

	//Remove duplicate elements from the sorted vector
	a.erase(std::unique(a.begin(), a.end()), a.end());
}

vector<int> ImageRetriever::computeMostRelevantByTfidf(const cv::Mat& desc){
	vector<OrderAux> orderVec;
	for (unsigned i = 0; i < Constants::VOCABULARY_NUM_WORDS; ++i){
		orderVec.push_back(OrderAux(i, desc.at<float>(0, i)*tfIdfPre[i]));
	}
	sort(orderVec.begin(), orderVec.end(), [](OrderAux order1, OrderAux order2){
		return order1.val > order2.val;
	});
	vector<int> ret;
	for (int i = 0; i < Constants::TFIDF_WORDS_CONSIDER; ++i){
		inplace_union(ret, invFileIndex[orderVec[i].pos]);
	}
	return ret;
}