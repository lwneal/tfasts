#include "Grid.h"
#include "Mask.h"

#include "dlib/threads.h"
#include "dlib/image_io.h"
#include "dlib/geometry.h"
#include "dlib/matrix.h"

#include <iostream>

using std::pair;
using std::vector;
using std::string;
using std::function;
using std::cerr;
using std::endl;

using dlib::point;

// Given values for superpixels, and a grid of those superpixels, return a new
// vector of values where each superpixel's value is the median of all sp's bordering it
vector<double> Grid::superpixel_median_filter(vector<double> &sp_labels) const {
	int nc = num_classes();
	dlib::matrix<bool> adjacency(nc, nc);
	adjacency = false;
	vector<double> median_values(sp_labels.size());
	// Generate adjacency matrix
	foreach([&](int x, int y) {
		adjacency( (*this)(x-1,y), (*this)(x,y) ) = true;
		adjacency( (*this)(x+1,y), (*this)(x,y) ) = true;
		adjacency( (*this)(x,y-1), (*this)(x,y) ) = true;
		adjacency( (*this)(x,y+1), (*this)(x,y) ) = true;
	});
	for (int s = 0; s < nc; s++) {
		// Vector of values of all adjacent superpixels
		vector<double> adjacent_vals;
		for (int i = 0; i < nc; i++) {
			if (adjacency(s, i)) {
				adjacent_vals.push_back(sp_labels[i]);
			}
		}
		std::sort(adjacent_vals.begin(), adjacent_vals.end());
		if (adjacent_vals.size() % 2 == 0) {
			median_values[s] = adjacent_vals[adjacent_vals.size() /2];
		}
		else {
			median_values[s] = 0.5 * (adjacent_vals[adjacent_vals.size()/2]
				+ adjacent_vals[1+adjacent_vals.size()/2]);
		}
	}
	return median_values;
}

// Given values for superpixels, and a grid of those superpixels, return a new
// vector of values where each superpixel's value is the median of all sp's bordering it
vector<double> Grid::superpixel_mean_filter(vector<double> &values) const {
	int nc = num_classes();
	dlib::matrix<bool> adjacency(nc, nc);
	vector<double> median_values(values.size());
	// Generate adjacency matrix
	foreach([&](int x, int y) {
		adjacency( (*this)(x-1,y), (*this)(x,y) ) = true;
		adjacency( (*this)(x+1,y), (*this)(x,y) ) = true;
	});
	for (int s = 0; s < nc; s++) {
		// Mean average of all adjacent superpixels
		double total = 0;
		int num = 0;
		for (int i = 0; i < nc; i++) {
			if (adjacency(s, i)) {
				total += values[i];
				num++;
			}
		}
		median_values[s] = total / num;
	}
	return median_values;
}

vector<vector<point>> Grid::get_segments() const {
	vector<vector<point>> segments(num_classes());
	foreach([&](int x, int y) {
		segments[at(x,y)].push_back(dlib::point(x,y));
	});
	return segments;
};

int Grid::replace_class(int old_class, int new_class) {
	int num_pix = 0;
	foreach([&](int x, int y) {
		if (at(x,y) == old_class) {
			at(x,y) = new_class;
			num_pix++;
		}
	}, 1);
	return num_pix;
}

int Grid::rebalance_labels() {
	vector<bool> class_exists(num_classes(), false);
	foreach([&](int x, int y) {
		class_exists[at(x,y)] = true;
	},1);
	for (int i = 0; i < class_exists.size(); i++) {
		// If there's an unused class index, replace the highest class with it
		if (class_exists[i] == false) {
			class_exists[i] = class_exists.back();
			class_exists.pop_back();
			int num_px = replace_class(class_exists.size(), i);
			i--;
		}
	}
	return class_exists.size()-1;
}

void Grid::foreach_worker(int start, int end, grid_fn_t fn) const {
	for (int yrow = start; yrow < end; yrow++) {
		for (int xcol = 0; xcol < width(); xcol++) {
			fn(xcol,yrow);
		}
	}
}

void Grid::initialize_worker(int start, int end, grid_initfn_t fn) {
	for (int yrow = start; yrow < end; yrow++) {
		for (int xcol = 0; xcol < width(); xcol++) {
			at(xcol,yrow) = fn(xcol,yrow);
		}
	}
}

void Grid::foreach(grid_fn_t fn, int num_threads) const {
	dlib::thread_pool pool(num_threads);
	for (int i = 0; i < num_threads; i++) {
		int start = i*height() / num_threads;
		int end = (i+1)*height() / num_threads;
		pool.add_task_by_value([=](){
			foreach_worker(start,end,fn);
		});
	}
	pool.wait_for_all_tasks();
}

// The bounds checking slows access considerably
int& Grid::operator()(int xcol, int yrow) {   
	clamp_val(xcol, 0, width()-1);
	clamp_val(yrow, 0, height()-1);
	return (*data)[yrow][xcol];
}
int Grid::operator()(int xcol, int yrow) const {   
	clamp_val(xcol, 0, width()-1);
	clamp_val(yrow, 0, height()-1);
	return (*data)[yrow][xcol];
}

int& Grid::at(int xcol, int yrow) {
	return (*data)[yrow][xcol];
}
int Grid::at(int xcol, int yrow) const {
	return (*data)[yrow][xcol];     
}

// bmp load constructor
Grid::Grid(const std::string &fname, int thresh):
	data(new dlib::array2d<int>)
{
	dlib::load_bmp(*data, fname);
	foreach([&](int x, int y) {
		at(x,y) = (at(x,y) > thresh)? 1 : 0;
	});
}

Grid::Grid(int width, int height, grid_initfn_t fn, int num_threads):
	data(new dlib::array2d<int>(height,width)) 
{
	dlib::thread_pool pool(num_threads);
	for (int i = 0; i < num_threads; i++) {
		int start = i*height / num_threads;
		int end = (i+1)*height / num_threads;
		pool.add_task_by_value([=](){
			initialize_worker(start,end,fn);
		});
	}
	pool.wait_for_all_tasks();
}

// Copy constructor
Grid::Grid(const Grid &copy):
	data(new dlib::array2d<int>(copy.height(),copy.width())) 
{
	copy.foreach([&](int x, int y) {
		at(x,y) = copy(x,y);
	});
}

int Grid::num_classes() const
{
	int max_val = -1;
	foreach([&](int x, int y) {
		if (at(x,y) > max_val)
			max_val = at(x,y);
	});

/*	// Just a test here for debugging.
	vector<bool> class_seen(max_val+1, false);
	for (int x = 0; x < width(); x++) {
		for (int y = 0; y < height(); y++) {
			assert(at(x,y) >= 0);
			assert(at(x,y) <= max_val);
			class_seen[at(x,y)] = true;
		}
	}
	for (int i = 0; i < class_seen.size(); i++)
		assert(class_seen[i] == true); */

	return max_val + 1;
}

Grid Grid::merge_small_segments(int min_size) const {
	Grid merged(*this);
	vector<vector<point>> segments = get_segments();
	for (int s = 0; s < segments.size(); s++) {
		// Is this segment too small?
		if (segments[s].size() < min_size) {
			// We must find the nearest large segment and merge with it
			point p = segments[s].front();
			int r = 1;
			int merge_with = 0;
			while (r < height()) {
				merge_with = merged(p.x()+r,p.y());
				if (segments[merge_with].size() >= min_size) break;
				merge_with = merged(p.x()-r,p.y());
				if (segments[merge_with].size() >= min_size) break;
				merge_with = merged(p.x(),p.y()+r);
				if (segments[merge_with].size() >= min_size) break;
				merge_with = merged(p.x()-r,p.y()-r);
				if (segments[merge_with].size() >= min_size) break;
				r += 1;
			}
			merged.replace_class(s, merge_with);
		}
	}
	merged.rebalance_labels();
	return merged;
}

Grid Grid::remove_below_thresh(Mask &mask, double thresh) const {
	Grid const &seg = *this;
	int num_classes = seg.num_classes();
	vector<double> energy(num_classes);
	vector<int> num_pixels(num_classes);
	seg.foreach([&](int x, int y) {
		int idx = seg.at(x,y);
		energy[idx] += mask.at(x,y);
		num_pixels[idx]++;
	});
	vector<bool> keep_seg(num_classes);
	for (int i = 0; i < num_classes; i++)
		keep_seg[i] = (energy[i]/num_pixels[i]) > thresh;
	Grid ret(width(), height(), [&](int x, int y) {
		return keep_seg[seg.at(x,y)] ? seg.at(x,y) : 0;
	});
	ret.rebalance_labels();
	return ret;
}

Grid Grid::split_connected_components() const {
	Grid split(width(), height(), [&](int x, int y) -> int {
		int val = at(x,y);
		return val;
	});	
	int init_num_classes = split.num_classes();
	cerr << "Starting split on " << init_num_classes << " classes" << endl;
	vector<bool> classes_seen(init_num_classes, false);
	dlib::matrix<bool> px_seen(width(), height());
	px_seen = false;
	int new_num_classes = init_num_classes;
	
	for (int x = 0; x < width(); x++) {
		for (int y = 0; y < height(); y++) {
			// Has this pixel been scanned already?
			if (px_seen(x,y) == true) continue;
			
			// This is either an unseen new component, or a duplicate of a previously seen
			int label = at(x,y);
			int newlabel = label;
			// Is this part of a connected component with a class label we've already seen?
			if (classes_seen[label]) {
				// If so, create a new class for this component of this segment
				newlabel = new_num_classes;
				new_num_classes++;
			}
			classes_seen[label] = true;
			// Find this pixel's connected component, set the 'seen' and newlabel
			vector<point> segment;
			segment.push_back(point(x,y));
			while (segment.size() > 0) {
				point p = segment.back();
				segment.pop_back();
				if (p.x() > 0 && px_seen(p.x()-1,p.y()) == false && split(p.x()-1,p.y()) == label)
					segment.push_back(point(p.x()-1,p.y()));
				if (p.x() < width()-1 && px_seen(p.x()+1,p.y()) == false && split(p.x()+1,p.y()) == label)
					segment.push_back(point(p.x()+1,p.y()));
				if (p.y() > 0 && px_seen(p.x(),p.y()-1) == false && split(p.x(),p.y()-1) == label)
					segment.push_back(point(p.x(),p.y()-1));
				if (p.y() < height()-1 && px_seen(p.x(),p.y()+1) == false && split(p.x(),p.y()+1) == label)
					segment.push_back(point(p.x(),p.y()+1));
				// We have now 'covered' this pixel
				px_seen(p.x(),p.y()) = true;
				// New class label, if appropriate
				split.at(p.x(),p.y()) = newlabel;
			}
		}
	}
	new_num_classes = split.rebalance_labels();
	cerr << "Split into " << new_num_classes << " connected components " << endl;
	
	return split;
}

Grid Grid::merge_island_segments() const {
	Grid labels(*this);
	int classes = num_classes();
	cerr << "Starting merge on " << classes << " segments" << endl;
	// Merge all segments surrounded entirely by other segments
	bool did_merge = false;
	do {
		did_merge = false;
		// Generate adjacency matrix
		dlib::matrix<bool> adjacent;
		adjacent.set_size(classes, classes);
		adjacent = false;
		foreach([&](int x, int y) {
			if (labels(x-1,y) != labels(x,y))
				adjacent(labels(x-1,y), labels(x,y)) = true;
			if (labels(x+1,y) != labels(x,y))
				adjacent(labels(x+1,y), labels(x,y)) = true;
			if (labels(x,y-1) != labels(x,y))
				adjacent(labels(x,y-1), labels(x,y)) = true;
			if (labels(x,y+1) != labels(x,y))
				adjacent(labels(x,y+1), labels(x,y)) = true;
		},1);

		// Find each segment with only 1 neighbor
		for (int i = 0; i < classes; i++) {
			int num_neighbors = 0;
			int last_neighbor = 0;
			for (int j = 0; j < classes; j++) {
				if (adjacent(i,j)) {
					num_neighbors++;
					last_neighbor = j;
				}
			}
			if (num_neighbors == 1) {
				labels.foreach([&](int x, int y) {
					if (labels.at(x,y) == i)
						labels(x,y) = last_neighbor;
				});
				did_merge = true;
			}
		}
	} while (did_merge);

	labels.rebalance_labels();
	cerr << "Now have " << labels.num_classes() << " segments" << endl;
	return labels;
}

Grid Grid::predicted_segmentation(
		std::vector<double> &labels_fg, 
		dlib::matrix<double> &labels_merger,
		double foreground_theta,
		double merger_delta) const
{
	// Create a grid with opposite-valued segment labels
	int num_sp = num_classes();
	Grid seg(width(), height(), [&](int x, int y) {
		return -1 * at(x,y);
	});
	
	// For each superpixel, 
	int current_class_label = 1;
	vector<bool> seen(num_sp, false);
	auto adjacent = get_adjacency();
	for (int s = 0; s < num_sp; s++) {
		// Is this SP just background, or already-covered foreground?
		if (seen[s] || (labels_fg[s] < foreground_theta)) {
			seg.replace_class(-s, 0);
			continue;
		}
		seen[s] = true;
		// This SP is part of a segment! DFS to find the rest of the segment
		vector<int> seg_sps;
		seg_sps.push_back(s);
		
		while (seg_sps.size() > 0) {
			int u = seg_sps.back();
			seg.replace_class(-1 * u, current_class_label);
			seg_sps.pop_back();
			for (int t = 0; t < num_sp; t++) {
				if (u != t && adjacent(u,t) && !seen[t]
					&& labels_fg[t] > foreground_theta
					&& 0.5 * (labels_merger(u,t) + labels_merger(t,u)) > merger_delta )
				{
					seg_sps.push_back(t);
					seen[t] = true;
				}
			}
		}
		current_class_label++;
	}
	// At this point every pixel in seg should be 0 or some class label
	for (int x = 0; x < seg.width(); x++)
		for (int y = 0; y < seg.height(); y++)
			assert(seg.at(x,y) >= 0);

	return seg;
}

// Given a segmentation and layer set, returns a matrix containing each superpixel's
//	average value for each layer (ie. the average of all pixels in that superpixel)
// Each row represents one superpixel
// Each column represents one layer
dlib::matrix<double> Grid::region_descriptors(vector<Mask> &layers) const
{
	// Sum up the region descriptor at each layer
	dlib::matrix<double> layer_vals(num_classes(), layers.size());
	layer_vals = 0;

	for (int layer = 0; layer < layers.size(); layer++) {
		foreach([&](int x, int y) {
			layer_vals(at(x,y), layer) += layers[layer].at(x,y);
		});
	}
	// Normalize for segment size (number of pixels)
	vector<int> sizes = class_sizes();
	
	for (int r = 0; r < sizes.size(); r++)
		for (int c = 0; c < layer_vals.nc(); c++)
			layer_vals(r,c) /= sizes[r];

	return layer_vals;
}

// Given a segmentation and layer set, returns a matrix containing each superpixel's
//	average value for each layer (ie. the average of all pixels in that superpixel),
//	and each superpixel's variance for each layer
// Each row represents one superpixel
// Each column represents one layer
dlib::matrix<double> Grid::region_descriptors_with_variance(vector<Mask> &layers) const
{
	// We make a matrix with 2 values per layer/feature/descriptor type
	int num_ftr = layers.size();
	vector<int> num_pixels = class_sizes();
	dlib::matrix<double> layer_vals(num_classes(), num_ftr * 2);
	layer_vals = 0;

	for (int layer = 0; layer < num_ftr; layer++) {
		foreach([&](int x, int y) {
			double val = layers[layer].at(x,y);
			// Keep track of sums in the left side of the matrix
			layer_vals(at(x,y), layer) += val;
			// Keep track of the sum of squared values in the right half of the matrix
			layer_vals(at(x,y), layer + num_ftr) += val*val;
		});
	}
	// Normalize for segment size (number of pixels) and use the right-hand side 
	//	squared values to compute variances
	for (int r = 0; r < layer_vals.nr(); r++) {
		for (int c = 0; c < num_ftr; c++) {
			double sum = layer_vals(r,c);
			double mean = sum / num_pixels[r];
			double sum_sqr = layer_vals(r,c+num_ftr);
			
			layer_vals(r,c) = mean;
			layer_vals(r,c+num_ftr) = (sum_sqr - (sum*mean)) / num_pixels[r];
		}
	}

	return layer_vals;
}

// This function assumes the grid's values are well-formed. Always make sure
//	that rebalance_labels() is called after any changes to a segmentation
vector<int> Grid::class_sizes() const 
{
	vector<int> sizes(num_classes());
	foreach([&](int x, int y) {
		sizes[at(x,y)]++;
	});
	for (int i = 0; i < sizes.size(); i++) {
		assert(sizes[i] > 0);
	}
	return sizes;
}

// This function counts how many pixels on each segment are labeled 'true', and
//	assigns each segment a truthiness value between 0 and 1
// It would be reasonable to consider all segments with truthiness > 0.5 to be
//	positive examples, and all others negative examples
vector<double> Grid::label_from_truth(Grid &truth) const
{
	// TODO: Either bounds-check/clamp all the time, or re-scale grids
	//	Some masks have a width of 935 instead of 936, and problems ensue
	//assert(truth.width() == width() && truth.height() == height());

	vector<int> sizes = class_sizes();
	vector<int> positives(sizes.size(), 0);

	foreach([&](int x, int y) {
		positives[at(x,y)] += (truth(x,y) > 0) ? 1 : 0;
	});

	vector<double> proportion_positive(sizes.size());
	for (int i = 0; i < sizes.size(); i++) {
		proportion_positive[i] = (double)(positives[i])/sizes[i];
	}
	return proportion_positive;
}

// Return a matrix specifying whether segment X is part of the same ground-truth segment
//	as segment Y, for all (X,Y) pairs
dlib::matrix<double> Grid::adjacent_merger_from_truth(Grid &truth) const {
	int nc = num_classes();
	dlib::matrix<double> adjacency(nc, nc);
	adjacency = 0;
	// Every segment should be adjacent to itself. Any two segments that border along a
	//	positive-labeled region are adjacent.
	foreach([&](int x, int y) {
		if (truth(x,y) > 0 && truth(x+1,y) > 0) {
			adjacency( (*this)(x+1,y), (*this)(x,y) ) = 1;
			adjacency( (*this)(x,y), (*this)(x+1,y) ) = 1;
		}
		if (truth(x,y) > 0 && truth(x,y+1) > 0) {
			adjacency( (*this)(x,y+1), (*this)(x,y) ) = 1;
			adjacency( (*this)(x,y), (*this)(x,y+1) ) = 1;
		}
	});
	return adjacency;
}

dlib::matrix<bool> Grid::get_adjacency() const {
	int nc = num_classes();
	dlib::matrix<bool> adjacency(nc, nc);
	adjacency = false;
	// Every segment should be adjacent to itself. Any two segments that border along a
	//	positive-labeled region are adjacent.
	foreach([&](int x, int y) {
		adjacency( (*this)(x-1,y), (*this)(x,y) ) = true;
		adjacency( (*this)(x+1,y), (*this)(x,y) ) = true;
		adjacency( (*this)(x,y-1), (*this)(x,y) ) = true;
		adjacency( (*this)(x,y+1), (*this)(x,y) ) = true;
	});
	return adjacency;
}

// Returns bounding boxes in -X -Y +X +Y format
dlib::matrix<dlib::rectangle> Grid::get_bounding_boxes() const
{
	dlib::matrix<dlib::rectangle> boxes(num_classes(), 1);
	boxes = dlib::rectangle(INT_MAX, INT_MAX, -INT_MAX, -INT_MAX);

	foreach([&](int x, int y) {
		boxes(at(x,y)) += dlib::point(x,y);
	});
	return boxes;
}

dlib::matrix<double> Grid::histogram_of_gradients(int num_buckets, Mask &hgrad, Mask &vgrad) const 
{
	const double epsilonsq = .000001;
	dlib::matrix<double> hog(num_classes(), num_buckets);
	hog = 0;

	foreach([&](int x, int y) {
		int sp_idx = (*this)(x,y);
		
		if (vgrad(x,y)*vgrad(x,y) > epsilonsq 
			&& hgrad(x,y)*hgrad(x,y) > epsilonsq
			&& sp_idx > 0) 
		{
			int bin = num_buckets * atan2(vgrad(x,y), hgrad(x,y));
			if (bin < 0 || bin >= num_buckets) bin = 0;
			hog(sp_idx,bin) += 1.0;
		}
	});
	for (int r = 0; r < hog.nr(); r++) {
		double sum = 0;
		for (int c = 0; c < hog.nc(); c++)
			sum += hog(r,c);
		if (sum < epsilonsq)
			continue;
		for (int c = 0; c < hog.nc(); c++)
			hog(r,c) /= sum;
	}
	return hog;
}