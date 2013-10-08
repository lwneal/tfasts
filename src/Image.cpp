#include "Image.h"
#include "Utility.h"
#include "Mask.h"
#include "Grid.h"

// Create a blank (black) image
Image::Image(int width, int height, dlib::rgb_pixel background):
	data(new dlib::array2d<dlib::rgb_pixel>(height, width))
{
}

// Load color image from file
Image::Image(std::string filename):
	data(new dlib::array2d<dlib::rgb_pixel>)
{
	dlib::load_image(*data, filename);
}

// Turn mask into image
Image::Image(const Mask &mask):
	data(new dlib::array2d<dlib::rgb_pixel>(mask.height(), mask.width()))
{
	double maxval = mask.get_max();
	assert(maxval > 0);
	mask.foreach([&](int x, int y) {
		float val = 255.0 * mask.at(x, y) / maxval;
		(*data)[y][x] = dlib::rgb_pixel(val, val, val);
	});
}

// Turn grid into image
Image::Image(const Grid &grid):
	data(new dlib::array2d<dlib::rgb_pixel>(grid.height(), grid.width()))
{
	grid.foreach([&](int x, int y) {
		float val = grid.at(x, y) * 100;
		(*data)[y][x] = dlib::rgb_pixel(val, val, val);
	});
}

bool Image::save(const std::string& filename) const {
        dlib::save_bmp(*data, filename);
	return true;
}
