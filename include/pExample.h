#pragma once

#include <vector>
using namespace std;

// this class stores an example for use in a classifier (e.g. random forest, etc).
class pExample
{
public:
	vector<float> featureVector_;	// the feature vector corresponding to this example
	int classLabel_;				// the class label for this example
	
	// empty constructor
	pExample() {}
	
	// constructor taking a feature vector and a label. the feature vector is copied
	pExample(vector<float>& featureVector, int classLabel) :
		featureVector_(featureVector), classLabel_(classLabel) {}
};