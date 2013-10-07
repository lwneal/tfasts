#pragma once
#include "Grid.h"
#include "Mask.h"

#include "dlib/image_io.h"
#include "dlib/dir_nav.h"

#include <string>
#include <memory>
#include <vector>
#include <functional>

typedef std::shared_ptr<dlib::array2d<dlib::rgb_pixel>> img_data_t;

// Forward declarations
class Grid;
class Mask;

// The Image class provides easy manipulation, input, and output
// of images, made from Masks, Grids, or loaded from file.
// Images are stored as 8/8/8 RGB (less precise than Mask's floats)

class Image {
	img_data_t data;

public:
	// Create a blank or solid-color image
	Image(int width, int height, dlib::rgb_pixel background = dlib::rgb_pixel(0,0,0));

	// Load color image from file
	Image(std::string filename);

	// Turn mask into image
	Image(const Mask &mask);

	int width() const {   return data->nc();   }
	int height() const {  return data->nr();   }
	long size() const {  return width()*height(); }

	// Save this image as a BMP
	bool save(const std::string& filename) const;
};
