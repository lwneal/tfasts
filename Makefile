CC=clang++ -std=c++11 -stdlib=libc++
CFLAGS=-g

all: Mask.o Grid.o pRandomForest.o
	${CC} ${CFLAGS} *.o src/spectrogram.cpp -I include -o bin/spectrogram

Mask.o:
	${CC} ${CFLAGS} -I include -c src/Mask.cpp

Grid.o:
	${CC} ${CFLAGS} -I include -c src/Grid.cpp

FFT.o:
	${CC} ${CFLAGS} -I include -c src/FFT.cpp

pRFDecisionTree.o:
	${CC} ${CFLAGS} -I include -c src/pRFDecisionTree.cpp

pRandomForest.o: pRFDecisionTree.o FFT.o
	${CC} ${CFLAGS} -I include -c src/pRandomForest.cpp

clean:
	rm -rf *.o bin/*
