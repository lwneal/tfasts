#pragma once
#include "Utility.h"

#include "dlib/array2d.h"
#include "dlib/threads.h"
#include "dlib/matrix.h"
#include "dlib/geometry.h"

#include <string>
#include <memory>
#include <vector>
#include <functional>

// We pass functions into several methods- usually lambdas
typedef std::shared_ptr<dlib::array2d<int>> grid_data_t;
typedef std::function<void (int,int)> grid_fn_t;
typedef std::function<int (int,int)> grid_initfn_t;

// Forward declarations
class Mask;

class Grid {
	grid_data_t data;

	void foreach_worker(int start, int end, grid_fn_t fn) const;
	void initialize_worker(int start, int end, grid_initfn_t fn);

	std::vector<std::vector<dlib::point>> get_segments() const;
	int replace_class(int old_class, int new_class);


public:

	// Load binary Grid (two classes, 1 and 0) from black/white bmp file
	Grid(const std::string &fname, int thresh = 0);

	// Normal init constructor, takes lambda to set values
	Grid(int width, int height, grid_initfn_t fn, 
		int num_threads = num_cores);

	// Copy constructor
	Grid(const Grid &copy);

	void foreach(grid_fn_t fn, int num_threads = num_cores) const;

	int width() const {   return data->nc();   }
	int height() const {  return data->nr();   }
	long size() const {  return width()*height(); }
	grid_data_t get_data() { return data; }

	// Get the highest-numbered label in this grid
	int num_classes() const;

	// Remove all empty labels
	int rebalance_labels();

	// Use this if you want to clamp to edges (eg when convolving)
	int& operator()(int xcol, int yrow);
	int operator()(int xcol, int yrow) const;

	// Use this for fast access with no bounds checking
	int& at(int xcol, int yrow);
	int at(int xcol, int yrow) const;

	// Grid where all small segments are merged
	Grid merge_small_segments(int min_size) const;

	// Removes (sets to 0) all segments with avg. mask value below thresh
	Grid remove_below_thresh(Mask &mask, double thresh) const;

	// Grid where each connected component is made into an individual segment
	Grid split_connected_components() const;

	// Merges all completely-surrounded segments with the one surrounding them
	Grid merge_island_segments() const;

	// Performs the Superpixel Merger segmentation based on the outputs of a FG/BG
	//	classifier and a pairwise merger classifier
	Grid predicted_segmentation(std::vector<double> &labels_fg, 
		dlib::matrix<double> &labels_merger, double foreground_theta,
		double merger_delta) const;

	// Given a set of Mask layers, get a matrix of averages for each class
	dlib::matrix<double> region_descriptors(std::vector<Mask> &layers) const;
	dlib::matrix<double> region_descriptors_with_variance(std::vector<Mask> &layers) const;
	
	// Get the size (number of pixels) of each class
	std::vector<int> class_sizes() const;

	// Given a ground-truth segmentation, generate labels for each class
	std::vector<double> label_from_truth(Grid &truth) const;
	// Generate true-false labels for connections between each pair of adjacent segments
	dlib::matrix<double> adjacent_merger_from_truth(Grid &truth) const;

	// Generate an adjacency matrix for classes/segments
	dlib::matrix<bool> get_adjacency() const;

	// Get bounding boxes for all segments
	dlib::matrix<dlib::rectangle> get_bounding_boxes() const;

	// Given a set of continuous labels for this grid's superpixels, find the median
	//	of labels of adjacent superpixels for each label.
	std::vector<double> superpixel_median_filter(std::vector<double> &sp_labels) const;
	std::vector<double> superpixel_mean_filter(std::vector<double> &sp_labels) const;

	// Each row represents one segment; there are num_classes() rows
	// Each column represents a range of angles; there are num_buckets columns
	// Each row is normalized to a sum of 1
	dlib::matrix<double> histogram_of_gradients(int num_buckets, Mask &hgrad, Mask &vgrad) const;
};