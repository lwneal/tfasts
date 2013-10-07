#include <string>
#include <iostream>

#include "pRandomForest.h"
#include <Mask.h>

using namespace std;


pRandomForest makeTree() {
  int num_trees = 10;
  int num_classes = 2;
  int feature_dim = 4;

  vector<pExample> training;
  pRandomForest *rf = new pRandomForest(num_trees, num_classes, feature_dim, training);
  return *rf;
}


int main(int argc, char *argv[]) {

  string msg("hello, world!");
  cout << msg << endl;

  return 0;
}
