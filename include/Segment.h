#pragma once
#include "dlib/geometry.h"

#include <assert.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <memory>

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::shared_ptr;


class Segment {

public:
	vector<pair<int,int>> points;

	int size() { return points.size(); }
	void add_point(int x, int y) {
		if (!contains(x,y)) {
			points.push_back(std::make_pair(x,y));
		}
	}
	void per_pixel(grid_scan_fn_t func) {
		for (int i = 0; i < points.size(); i++) {
			func(points[i].first, points[i].second);
		}
	}
	bool contains(int x, int y) {
		bool contains = false;
		per_pixel([&](int nx, int ny) { 
			if (x==nx && y==ny) contains = true;
		});
		return contains;
	}
	// TODO this is terrible and bad, rewrite it
	Segment get_perim(dlib::rectangle bounds, std::function<bool (int,int)> decision) {
		Segment perim;
		per_pixel([&](int x, int y) {
			if (x-1 >= bounds.left() && decision(x-1,y))
				perim.add_point(x-1,y);
			if (x+1 <= bounds.right() && decision(x+1,y))
				perim.add_point(x+1,y);
			if (y-1 >= bounds.top() && decision(x,y-1))
				perim.add_point(x,y-1);
			if (y+1 <= bounds.bottom() && decision(x,y+1))
				perim.add_point(x,y+1);
		});
		return perim;
	}
	void operator+=(Segment& rhs) {
		rhs.per_pixel([&](int x, int y) {
			add_point(x,y);
		});
	}
};

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