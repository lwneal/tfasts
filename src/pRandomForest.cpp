// NOTE: this is a modified version of my original random forest.
// the main difference is that tree leaves store a histogram instead of a majority class

#include "pRandomForest.h"

using namespace std;

pRandomForest::pRandomForest(int numTrees, int numClasses, int featureDim, 
	vector<pExample>& trainingExamples): num_splits(featureDim, 0), avg_gini(featureDim, 0)
{
	numClasses_ = numClasses;
		
	// the number of features to decide between in each node of the decision trees.
	int numFeaturesToTry = (int)pVectorUtils::log2(featureDim) + 1;
	//cout << "numFeaturesToTry = " << numFeaturesToTry << endl;
		
	// just for code clarity, give a name to the number of examples
	int numExamples = trainingExamples.size();
		
	// For analysis of features
	pRFDecisionTree::setAnalysisRF(this);

	// build the decision tree ensemble
	for(int i = 0; i < numTrees; ++i)
	{
		cout << "*"; cout.flush();
			
		// construct a different bootstrap sample from the training set for each tree
		vector<pExample*> bootstrappedExamples;
		for(int j = 0; j < numExamples; ++j)
		{
			int randomIndex = rand() % numExamples;
			bootstrappedExamples.push_back(&trainingExamples[randomIndex]);
		}
			
		// make a tree from this bootstrapped sample and add it to the ensemble
		//cout << "About to build tree " << i << " with " << bootstrappedExamples.size() << endl;
		pRFDecisionTree* tree = new pRFDecisionTree(bootstrappedExamples, numClasses, featureDim, numFeaturesToTry);
			
		trees_.push_back(tree);
	}
	//cout << endl;
}
	
// add up the output histograms for each tree
vector<float> pRandomForest::estimateClassProbabilities(vector<float>& feature)
{
	vector<float> outClassProbabilities(numClasses_);
		
	for(int i = 0; i < (int)trees_.size(); ++i)
	{
		vector<float> histo = trees_[i]->getClassHistogramForInput(feature);
			
		for(int c = 0; c < numClasses_; ++c)
			outClassProbabilities[c] += histo[c];
	}
		
	pVectorUtils::normalizeVectorToPDF(outClassProbabilities);
		
	return outClassProbabilities;
}
	
	
// predict the class label for a feature
int pRandomForest::classify(vector<float>& feature)
{
	vector<float> classProbabilities = estimateClassProbabilities(feature);
	return pVectorUtils::argmax(classProbabilities);
}
		
pRandomForest::~pRandomForest()
{
	for(int i = 0; i < (int)trees_.size(); ++i)
		delete trees_[i];
}
	
void pRandomForest::save(string filename)
{
	pTextFile f(filename, PFILE_WRITE);
		
	/*
	f << numClasses_ << endl;
		
	for(int i = 0; i < (int)trees_.size(); ++i)
	{
		f << trees_[i]->toString() << endl;
	}
	*/
		
	f.close();
}
	
void pRandomForest::load(string filename)
{
	pTextFile f(filename, PFILE_READ);
		
	numClasses_ = pStringUtils::stringToInt(f.readLine());
		
	while(!f.eof())
	{
		string line = f.readLine();
		trees_.push_back(new pRFDecisionTree(line, numClasses_));
	}
		
	f.close();
}
