
all: 
	g++ src/spectrogram.cpp src/pRandomForest.cpp src/pRFDecisionTree.cpp -lstdc++ -I include -o bin/spectrogram
