// NOTE: this is a modified version of my standard random forest.
// The main difference is that leaves store a histogram instead of a single majority class.

#pragma once

#include "pExample.h"
#include "pVectorUtils.h"
#include "pRandom.h"
#include "pStringUtils.h"
#include "pRandomForest.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class pRandomForest;

// used to sort pExamples on a particular feature
class pExampleComparator
{
public:
	pExampleComparator(int indexToCompare) : indexToCompare_(indexToCompare) {}
	
	int indexToCompare_;
	
	bool operator ()(const pExample* a, const pExample* b) 
	{
		return a->featureVector_[indexToCompare_] < b->featureVector_[indexToCompare_];
	}
};

// this class implements the CART algorithm for decision tree learning, with minor modifications for use with Random Forest
// multiple classes are supportd, and all features are assumed to be continuous
class pRFDecisionTree
{
	// For debugging and analysis
	static pRandomForest *prf;
public:
	bool isLeaf_;						// nodes are either internal (i.e. test a condition such as x10 < 5.3), or leaves (which are labeled with a class)

	vector<float> labelHistogram_;		// if this node is a leaf, it stores a histogram over classes
	
	int testFeature_;					// if this node is internal, this is the feature being tested
	float testThreshold_;				// if this node is internal, this is the threshold against which the tested feature is compared
	pRFDecisionTree* lessSubtree_;		// if this node is internal, this is the subtree for examples with attribute less than the test threshold
	pRFDecisionTree* moreSubtree_;		// if this node is internal, this is the subtree for examples with attribute greater than or equal to the test threshold

	// We set this once before experimentation, then check num_splits and avg_gini
	//	in the target pRandomForest
	static void setAnalysisRF(pRandomForest *p) {
		prf = p;
	}
		
	// test if all of the labels passed in are the same
	static bool allSameClass(vector<pExample*>& examples);
	
	// split the examples into outExamplesLess and outExamplesGreaterEq by comparing them on splitFeature against splitThreshold
	static void splitExamples(vector<pExample*>& examples, int splitFeature, float splitThreshold, vector<pExample*>& outExamplesLess, vector<pExample*>& outExamplesGreaterEq);
	
	// measure the gini impurity in the class labels for a collection of examples
	// profiling indicates that a lot of time is spent in this function, so it is optimized
	static float gini(vector<int>& classCounts, int numExamples, int numClasses);
	
	static vector<float> classHistogram(vector<pExample*>& examples, int numClasses);

	// grow a tree from a labeled training data set, following the CART algorithm, with Gini impurity, and the following modification for Random Forest:
	// rather than chosing the best split variable amongst all of them, select a random subset. numFeaturesToTry is the number of features in this subset.
	pRFDecisionTree(vector<pExample*>& examples, int numClasses, int featureDim, 
		int numFeaturesToTry, int depth = 0);
	
	void makeLeaf(vector<pExample*>& examples, int numClasses);

	// return the class label for feature vector as determined by this decision tree
	int classify(vector<float>& featureVector);
	
	// given an input feature vector, find the corresponding leaf
	// and return the class histogram it stores
	vector<float> getClassHistogramForInput(vector<float>& featureVector);
	
	// take a histogram, find the majority class, make that probability 1 and all others 0
	void makeHistogramIntoMajority(vector<float>& hist);
	
	// deconstructor recusrively deletes children
	~pRFDecisionTree();
	
	//// file io ////
	// return a string representation of this tree
	string toString();
	
	// parse a decision tree from a saved string
	pRFDecisionTree(string src, int numClasses);
	
};