#pragma once
#include <functional>

#include "dlib/geometry.h"
#include "dlib/dir_nav.h"

// How many threads should we run at a time?
const int num_cores = 4;

template <typename matrix_t>
inline void foreach(matrix_t &object, std::function<void (int,int)> fn)
{
	for (int xcol = 0; xcol < object.width(); xcol++) {
		for (int yrow = 0; yrow < object.height(); yrow++) {
			fn(xcol,yrow);
		}
	}
}

inline void foreach(dlib::rectangle rect, std::function<void (int,int)> fn)
{
	for (int xcol = rect.left(); xcol < rect.right(); xcol++) {
		for (int yrow = rect.top(); yrow < rect.bottom(); yrow++) {
			fn(xcol,yrow);
		}
	}
}

// Uses dlib::directory to get all files from a directory, matching
//	a given extension. Removes paths and extensions
inline std::vector<std::string> get_short_filenames_from_dir(
	std::string dir, std::string ext) 
{
	std::vector<dlib::file> files;
	std::vector<std::string> file_names;
	dlib::directory directory(dir);

	directory.get_files(files);
	for (int i = 0; i < (int)files.size(); i++) {
		int idx = files[i].name().rfind(ext);
		if (ext == "*" || idx != std::string::npos) {
			std::string fname = files[i].name();
			file_names.push_back(fname.substr(0, idx));
		}
	}
	std::sort(file_names.begin(), file_names.end());
	return file_names;
}

// Uses dlib::directory to get all files from a directory, matching
//	a given extension. Returns full path names
inline std::vector<std::string> get_full_filenames_from_dir(
	std::string dir, std::string ext) 
{
	std::vector<dlib::file> files;
	std::vector<std::string> file_names;
	dlib::directory directory(dir);

	directory.get_files(files);
	for (int i = 0; i < (int)files.size(); i++) {
		int idx = files[i].name().rfind(ext);
		if (ext == "*" || idx != std::string::npos) {
			file_names.push_back(files[i].full_name());
		}
	}
	std::sort(file_names.begin(), file_names.end());
	return file_names;
}

// Clamp 'val' to the range [min,max]
template <typename T>
inline void clamp_val(T& val, T min, T max) {
	if (val < min)
		val = min;
	else if (val > max)
		val = max;
}

// Extend a dlib::rectangle to include the given point
// TODO: replace this with rect += dlib::point(x,y);
inline void rect_extend(int x, int y, dlib::rectangle &rect) {
	if (x < rect.left())
		rect.set_left(x);
	if (x > rect.right())
		rect.set_right(x);
	if (y < rect.top())
		rect.set_top(y);
	if (y > rect.bottom())
		rect.set_bottom(y);
}

inline void normalizeVectorToPDF(std::vector<double>& w)
{
	double sum = 0;
	for(int i = 0; i < (int)w.size(); ++i) 
		sum += w[i];
	assert(sum != 0);
	for(int i = 0; i < (int)w.size(); ++i) 
		w[i] /= sum;
}
	
inline double giniForDistribution(std::vector<double>& prob)
{
	double gini = 1;
	for(int i = 0; i < (int)prob.size(); ++i) 
		gini -= prob[i] * prob[i];
	return gini;
}
