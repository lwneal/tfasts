#pragma once
#include "Utility.h"

#include "dlib/array2d.h"
#include "dlib/geometry.h"

#include <string>
#include <memory>
#include <vector>
#include <functional>

// We pass functions into several Mask methods- usually lambdas
typedef std::shared_ptr<dlib::array2d<double>> mask_data_t;
typedef std::function<void (int,int)> pixel_fn_t;
typedef std::function<double (int,int)> pixel_initfn_t;

// Forward declarations
class Grid;

// Mask is the class that holds our primary data structure: a 2D
//	array of floating-point values. Spectrograms are masks, where
//	each element (or 'pixel') represents audio amplitude.
class Mask {
	mask_data_t data;
	int sample_rate = -1;

	void foreach_worker(dlib::rectangle rect, pixel_fn_t fn) const;
	void initialize_worker(int start, int end, pixel_initfn_t fn);

public:
	// Function to load a mask from a previously-output probability mask BMP
	Mask(std::string &bmp_in);
	
	// Spectrogram generation function. Specs are just masks.
	// Resultant height will be fft_size/2, and width will be
	// (number of samples) / (fft_step)
	Mask(std::string &wav_fn, int fft_size, int fft_step);

	// The idea here is that we generate each mask with some
	// per-pixel kernel function, which is applied in parallel
	Mask(int width, int height, pixel_initfn_t fn, 
		int num_threads = num_cores);

	// Copy constructor
	Mask(const Mask &copy);

	// Inverse FFT to save as a WAV audio file
	void save_wav(std::string &filename) const;

	// Allows a given function to iterate over all coordinates in
	//	this mask. Runs using as many threads as specified.
	void foreach(pixel_fn_t fn, int num_threads = num_cores) const;
	void foreach(pixel_fn_t fn, dlib::rectangle rect, 
		int num_threads = num_cores) const;

	int width() const {   return data->nc();   }
	int height() const {  return data->nr();   }
	long size() const {  return width()*height(); }
	mask_data_t get_data() const { return data; }

	// Use this if you want to clamp to edges (eg when convolving)
	double& operator()(int xcol, int yrow);
	double operator()(int xcol, int yrow) const;

	// Use this for fast access with no bounds checking
	double& at(int xcol, int yrow);
	double at(int xcol, int yrow) const;

	// Return the maximum valued pixel in the mask
	double get_max() const;

	// Returns the mean of all pixels in the mask
	double get_mean() const;

	// Divide each pixel by the given value
	Mask div_by(double div) const;

	// Multiply each pixel by the given value
	Mask mult_by(double mul) const;

	// Raise each pixel to the given power
	Mask raise_to(double pow) const;

	// Set maximum value to 1
	Mask norm_to_max() const {
		return div_by(get_max());
	}

	// Zeros pixels above/below the given frequencies
	void band_pass(int hi_pass_hz, int low_pass_hz=-1);

	// Set mean value to avg
	Mask norm_to_mean(double avg) const {
		return mult_by(avg / get_mean());
	}

	// Clamps values to a maximum of max_val
	Mask clamp_to(double max_val) const {
		return Mask(width(), height(), [&](int x, int y) {
			return at(x,y) > max_val ? max_val : at(x,y);
		});
	}

	// A mask where each pixel's value is simply its (scaled) x-coordinate
	// Useful for handling pixel position and attributes simultaneously
	// Scales such that 
	Mask get_xcoord() const {
		return Mask(width(), height(), 
			[&](int x, int y) { return x; });
	}
	Mask get_ycoord() const {
		return Mask(width(), height(), 
			[&](int x, int y) { return y; });
	}

	// Basically divides each row by the (root mean square) average of
	//  the 20% quietest frames (assuming the 20% is all background)
	// Hence, all the noise is (in theory) normalized to white noise
	Mask whitening_filter(double quietest_frames = 0.20) const;

	// Two-pass gaussian blur with given sigma. Clamps to edges.
	Mask gaussian_blur(double sigma = 2.0) const;

	// Outputs a mask where each pixel's value is the variance of all
	//	pixels in a <radius> size window in this mask, weighed by
	//	their distance from the pixel via a Gaussian kernel
	Mask variance_mask(int radius = 5, double sigma = 3.0) const;

	// Outputs a mask with each column normalized to a PDF (sums to 1)
	Mask normpdf() const;

	// Outputs an X-value and a Y-value mask, where values are the coordinates
	//	of the highest peak within a window around each pixel, but weighted
	//	by a gaussian kernel with given sigma. Gives a voronoi-like effect
	std::pair<Mask,Mask> voronoi_peaks(double sigma = 5.0) const;

	// Outputs a Grid where each segment corresponds to a connected component
	//	of pixels with value > thresh
	Grid threshold_segment(double thresh) const;

	// Calculates a 'noise profile' estimating the avg background noise
	//	level at each frequency bin, from lowest to highest.
	std::vector<double> noise_profile(double percent_lowest) const;

	// Returns the Sobel gradient of this mask (as a pair of masks)
	std::pair<Mask, Mask> sobel_gradient() const;
	// Returns the horizontal gradient of an image
	Mask sobel_gradient_x() const;
	// Returns the vertical gradient of an image
	Mask sobel_gradient_y() const;
};
