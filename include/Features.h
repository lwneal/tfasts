#pragma once

#include "Mask.h"
#include "Grid.h"

#include "dlib/geometry.h"

#include <assert.h>
#include <vector>
#include <iostream>

using std::vector;

Mask preprocess_spec_icassp(Mask &spec, int hi_pass_hz) {
	spec.band_pass(hi_pass_hz);

	spec = spec.whitening_filter();

	spec = spec.norm_to_max();

	spec.foreach([&](int x, int y) {
		spec(x,y) = sqrt(spec.at(x,y));
	});
	return spec;
}

vector<float> extract_feature_perpixel_icassp(Mask &mask, int x, int y) {
	vector<float> features(128);
	double total = 0, square_total = 0;
	for (int u = x - 4; u <= x + 4; u++) {
		for (int v = y - 4; v <= y + 4; v++) {
			float val = mask(u,v);
			total += val;
			square_total += val*val;
			features.push_back(val);
		}
	}
	features.push_back(square_total - (total*total)/features.size());
	features.push_back(y);
	return features;
}
