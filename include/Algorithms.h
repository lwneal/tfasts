#pragma once

#include "Mask.h"
#include "Grid.h"

#include "dlib/geometry.h"

#include <assert.h>
#include <vector>
#include <iostream>

using std::string;
using std::vector;
using std::cerr;
using std::endl;

// For each k-means superpixel segment, track its bounding box
//	and its centroid value. Note that the 'source' vector will
//	start with weight-scaled average X and average Y
struct SegmentCluster {
	int num_pixels;			// Number of pixels in this cluster
	dlib::rectangle box;	// Bounding box
	vector<double> value;	// Centroid value

	SegmentCluster(int x, int y, vector<Mask> &source):
		box(dlib::point(x,y)), num_pixels(1)
	{
		for (int i = 0; i < (int)source.size(); i++) {
			value.push_back(source[i](x,y));
		}
	}

	SegmentCluster(int x, int y, Mask &blurred, Mask &variance,
		Mask &gradient_x, Mask &gradient_y, Mask &peaks_x, Mask &peaks_y):
		box(dlib::point(x,y)), num_pixels(1)
	{
		// Value vector contains:
		// X and Y values
		value.push_back(double(x));
		value.push_back(double(y));
		// Blurred amplitude
		value.push_back(blurred(x,y));
		// Variance value
		value.push_back(variance(x,y));
		// Gradient values
		value.push_back(gradient_x(x,y));
		value.push_back(gradient_y(x,y));
		// Nearest-peak values
		value.push_back(peaks_x(x,y));
		value.push_back(peaks_y(x,y));
	}

	operator std::string() const {
		std::stringstream ss;
		ss << "Size: " << num_pixels << endl;
		ss << "Value: ";
		for (int i = 0; i < (int)value.size(); i++)
			ss << value[i] << "\t";
		ss << endl << "Box: " << box << endl;
		return ss.str();
	}
};


// Given a segment bounding box, generates a larger 'scan box' around
//	it, clamped to the bounds of the entire spectrogram/image
dlib::rectangle get_scan_box(
	const dlib::rectangle &seg_box, 
	const dlib::rectangle &bounds,
	int horizontal_extend,
	int vertical_extend)
{
	dlib::rectangle scan_area(
		seg_box.left() - horizontal_extend, 
		seg_box.top() - vertical_extend,
		seg_box.right() + horizontal_extend, 
		seg_box.bottom() + vertical_extend);
	return scan_area.intersect(bounds);
}

// Takes as input some number of Masks over a spectrogram or image
//	Assumes masks[0] contains the x-value of each pixel, and masks[1] y-val
//	masks[2] and upward can be anything- intensity, color channels, 
//	convolutions, variance, x/y values of nearest peaks, etc
// The vector 'weights' contains a constant weight applied to each
//	corresponding mask.
// kval is the (initial) number of pixels
// Outputs a superpixel-style clustering of pixels in the spectrogram
// Superpixel clustering works as follows:
//	Initialize K attractors, positioned uniformly across the image
//	At each iteration, (iterate 10 times or so)
//		Label each pixel with the index of the nearest attractor
//		Update each attractor to the mean of its pixels' values
// The trick is to only search locally when assigning pixels to attractors
Grid local_kmeans_clustering(
	vector<Mask> &in_masks, 
	vector<double> &weights, 
	int kval, 
	bool enforce_contiguity = true) 
{
	vector<Mask> masks;
	// Instead of multiplying weights every time, scale all masks once
	for (int i = 0; i < (int)in_masks.size(); i++)
		masks.push_back(in_masks[i].mult_by(weights[i]));

	int width = masks.front().width();
	int height = masks.front().height();
	int depth = masks.size();
	dlib::rectangle bounds(0,0,width,height);
	vector<SegmentCluster> clusters;
	
	// We need one weight for each mask element
	assert(masks.size() == weights.size());
	// Masks must match in dimensions
	for (int i = 0; i < (int)masks.size(); i++)
		assert(masks[i].width() == width && masks[i].height() == height);

	// Initialize labels
	Grid labels(width, height, [&](int x, int y) { return 0; } );
	// This keeps track of each pixel's distance-to-current-closest-centroid
	Mask label_dist(width, height, [&](int x, int y) { return 0; });

	// Seed kval clusters across the image in a uniform grid
	double hw = (double)height / width;
	int rows = sqrt(kval*hw) + 0.5;
	for (int r = 0; r < rows; r++) {
		int row_yval = height * (double(r) + 0.5) / rows; 
		int first_in_row = (kval * r) / rows;
		int last_in_row = (kval * (r+1)) / rows;
		for (int c = first_in_row; c < last_in_row; c++) {
			int c_xval = width * (double(c-first_in_row) + 0.5) / (last_in_row-first_in_row);
			labels.at(c_xval, row_yval) = c;
			clusters.push_back(SegmentCluster(c_xval,row_yval,masks));
		}
	}

	// These determine how 'local' our local search is. If they're too small, some pixels
	//	could be missed
	int h_reach = width / sqrt(double(kval));
	int v_reach = height / sqrt(double(kval));

	// Now assign all pixels to the nearest cluster, re-average clusters, rinse and repeat
	for (int iter = 0; iter < 10; iter++) {
		// Clear distance-to-nearest-centroid mask
		label_dist = Mask(width, height, [&](int x, int y)  
			{ return std::numeric_limits<double>::max(); });

		// Assign each pixel to the centroid closest to it
		for (int c = 0; c < (int)clusters.size(); c++) {
			SegmentCluster& seg = clusters[c];
			dlib::rectangle scanb = get_scan_box(seg.box,bounds,h_reach,v_reach);
			label_dist.foreach([&](int x, int y) {
				double dist = 0;
				for (int d = 0; d < depth; d++) {
					double dim_d_dist = masks[d].at(x,y) - seg.value[d];
					dist += dim_d_dist*dim_d_dist;
				}
				dist = sqrt(dist);
				if (dist < label_dist.at(x,y)) {
					label_dist.at(x,y) = dist;
					labels.at(x,y) = c;
				}
			}, scanb);
		}

		// Update bounding boxes and pixel count
		for (int c = 0; c < (int)clusters.size(); c++) {
			clusters[c].box = dlib::rectangle(INT_MAX, INT_MAX, -INT_MAX, -INT_MAX);
			clusters[c].num_pixels = 0;
		}
		labels.foreach([&](int x, int y) {
			int c = labels.at(x,y);
			rect_extend(x,y,clusters[c].box);
			clusters[c].num_pixels++;
		});
			
		// Reset and update cluster centroids
		for (int c = 0; c < (int)clusters.size(); c++) {
			clusters[c].value = vector<double>(depth,0);
		}
		for (int d = 0; d < depth; d++) {
			labels.foreach([&](int x, int y) {
				clusters[labels.at(x,y)].value[d] += masks[d].at(x,y);
			});
		}
		for (int c = 0; c < (int)clusters.size(); c++) {
			for (int d = 0; d < depth; d++) {
				clusters[c].value[d] /= clusters[c].num_pixels;
			}
		}
	}
	labels.rebalance_labels();
	return labels;
}



Grid thesis_local_kmeans_clustering(
		int kval,		// Initial number of clusters to use
		double time_freq_weight,	// Distance measure weight given to spatial locality
		Mask blurred, double blur_value_weight,	// Weight given to gaussian blurred
		Mask variance, double variance_weight,	// Weight given to variance
		Mask gradient_h, Mask gradient_v, double gradient_weight,
		Mask peaks_h, Mask peaks_v, double peaks_weight)
{

	int width = blurred.width();
	int height = blurred.height();
	dlib::rectangle bounds(0,0,width,height);
		
	// Initialize labels
	vector<SegmentCluster> clusters;
	Grid labels(width, height, [&](int x, int y) { return 0; } );
	// This keeps track of each pixel's distance-to-current-closest-centroid
	Mask label_dist(width, height, [&](int x, int y) { return 0; });

	// Seed kval clusters across the image in a uniform grid
	double hw = (double)height / width;
	int rows = sqrt(kval*hw) + 0.5;
	for (int r = 0; r < rows; r++) {
		int row_yval = height * (double(r) + 0.5) / rows; 
		int first_in_row = (kval * r) / rows;
		int last_in_row = (kval * (r+1)) / rows;
		for (int c = first_in_row; c < last_in_row; c++) {
			int c_xval = width * (double(c-first_in_row) + 0.5) / (last_in_row-first_in_row);
			labels.at(c_xval, row_yval) = c;
			clusters.push_back(SegmentCluster(c_xval, row_yval, blurred, 
				variance, gradient_h, gradient_v, peaks_h, peaks_v));
		}
	}

	// These determine how 'local' our local search is. If they're too small, some pixels
	//	could be missed
	int h_reach = sqrt(double(width * height) / double(kval));
	int v_reach = h_reach;

	// Now assign all pixels to the nearest cluster, re-average clusters, rinse and repeat
	for (int iter = 0; iter < 10; iter++) {
		// Clear distance-to-nearest-centroid mask
		label_dist = Mask(width, height, [&](int x, int y)  
			{ return std::numeric_limits<double>::max(); });

		// Assign each pixel to the centroid closest to it
		for (int c = 0; c < (int)clusters.size(); c++) {
			SegmentCluster& seg = clusters[c];
			dlib::rectangle scanb = get_scan_box(seg.box,bounds,h_reach,v_reach);
			label_dist.foreach([&](int x, int y) {
				double time_diff = double(x) - seg.value[0];
				double freq_diff = double(y) - seg.value[1];
				double tf_dist = sqrt( time_diff*time_diff + freq_diff*freq_diff);
				double blur_dist = blurred.at(x,y) - seg.value[2];
				double var_dist = variance.at(x,y) - seg.value[3];
				double gh_diff = gradient_h.at(x,y) - seg.value[4];
				double gv_diff = gradient_v.at(x,y) - seg.value[5];
				double g_dist = sqrt(gh_diff*gh_diff + gv_diff*gv_diff);
				double peakh_diff = peaks_h.at(x,y) - seg.value[6];
				double peakv_diff = peaks_v.at(x,y) - seg.value[7];
				double peaks_dist = sqrt(peakh_diff*peakh_diff + peakv_diff*peakv_diff);

				double dist = time_freq_weight * tf_dist + blur_value_weight * blur_dist
					+ variance_weight * var_dist + gradient_weight * g_dist
					+ peaks_weight * peaks_dist;

				if (dist < label_dist.at(x,y)) {
					label_dist.at(x,y) = dist;
					labels.at(x,y) = c;
				}
			}, scanb);
		}

		// Update bounding boxes and pixel count
		for (int c = 0; c < (int)clusters.size(); c++) {
			clusters[c].box = dlib::rectangle(INT_MAX, INT_MAX, -INT_MAX, -INT_MAX);
			clusters[c].num_pixels = 0;
		}
		labels.foreach([&](int x, int y) {
			int c = labels.at(x,y);
			rect_extend(x,y,clusters[c].box);
			clusters[c].num_pixels++;
		});
			
		// Reset and update cluster centroids
		for (int c = 0; c < (int)clusters.size(); c++) {
			for (int d = 0; d < 8; d++)
				clusters[c].value[d] = 0;
		}
		labels.foreach([&](int x, int y) {
			clusters[labels.at(x,y)].value[0] += double(x);
			clusters[labels.at(x,y)].value[1] += double(y);
			clusters[labels.at(x,y)].value[2] += blurred.at(x,y);
			clusters[labels.at(x,y)].value[3] += variance.at(x,y);
			clusters[labels.at(x,y)].value[4] += gradient_h.at(x,y);
			clusters[labels.at(x,y)].value[5] += gradient_v.at(x,y);
			clusters[labels.at(x,y)].value[6] += peaks_h.at(x,y);
			clusters[labels.at(x,y)].value[7] += peaks_v.at(x,y);
		});
		for (int c = 0; c < (int)clusters.size(); c++) {
			for (int d = 0; d < 8; d++) {
				clusters[c].value[d] /= clusters[c].num_pixels;
			}
		}
	}
	labels.rebalance_labels();
	return labels;
}
