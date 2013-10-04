// This is a modified version of Forrest Briggs' implementation of the Random Forest
//	decision tree ensemble. 
// Interior nodes store one axis-aligned decision boundary each, and leaves store
//	a histogram of class labels.
// The tree is built by finding the optimal (best Gini coefficient) decision boundary
//	at each node, selecting from a random subset of features
#pragma once

#define RF_FEATURE_ANALYSIS true

#include "pVectorUtils.h"
#include "pRFDecisionTree.h"
#include "pTextFile.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

class pRFDecisionTree;

class pRandomForest
{
public:

	// For analysis, keep track of the # of splits, and the avg. Gini coefficient for those
	//	splits, for each feature.
	vector<int> num_splits;
	vector<double> avg_gini;

	// the ensemble of trees
	vector<pRFDecisionTree*> trees_;
	
	// the number of classes in the classification problem
	int numClasses_;

	// If RF_FEATURE_ANALYSIS, then this sets a global var in pRFDecisionTree
	// WARNING: Feature analysis is not threadsafe!
	pRandomForest(int numTrees, int numClasses, int featureDim, vector<pExample>& trainingExamples);
	
	// add up the output histograms for each tree
	vector<float> estimateClassProbabilities(vector<float>& feature);
	
	// predict the class label for a feature
	int classify(vector<float>& feature);
		
	~pRandomForest();
	pRandomForest() {}
	
	// Serialization and deserialization
	void save(string filename);
	void load(string filename);
};