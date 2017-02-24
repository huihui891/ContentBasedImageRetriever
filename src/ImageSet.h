#ifndef TESTSET_H
#define TESTSET_H

#include <string>
#include <vector>


class ImageSet {
public:

	ImageSet(int category, const std::string& path, unsigned first, unsigned count);


	std::vector<std::string> getImagesPaths() const;

	int getCategory() const;


private:
	
	const int category;
	std::vector<std::string> images;
	
};


#endif