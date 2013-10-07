CC=clang++ -std=c++11 -stdlib=libc++
CFLAGS=-g

all: 
	${CC} ${CFLAGS} src/spectrogram.cpp src/pRandomForest.cpp src/pRFDecisionTree.cpp -I include -o bin/spectrogram
