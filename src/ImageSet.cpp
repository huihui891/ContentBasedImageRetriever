#include "ImageSet.h"



ImageSet::ImageSet(int category, const std::string& path, unsigned first, unsigned count)
	: category(category)
{
	
	for (unsigned i = first; i < first+count; i++) {
		images.emplace_back(path + std::to_string(i) + ".jpg");
	}

}



std::vector<std::string> ImageSet::getImagesPaths() const {
	return images;
}


int ImageSet::getCategory() const {
	return category;
}