CC=clang++ -std=c++11 -stdlib=libc++
CFLAGS=-g

all: spectrogram

spectrogram: src/spectrogram.cpp Mask.o pRandomForest.o
	${CC} ${CFLAGS} *.o src/spectrogram.cpp -I include -o bin/spectrogram

Mask.o: include/Mask.h src/Mask.cpp dlib Grid.o
	${CC} ${CFLAGS} -I include -c src/Mask.cpp

Grid.o: include/Grid.h src/Grid.cpp dlib
	${CC} ${CFLAGS} -I include -c src/Grid.cpp

dlib: source.o
source.o: include/dlib/all/source.cpp
	${CC} ${CFLAGS} -DDLIB_NO_GUI_SUPPORT -I include -c include/dlib/all/source.cpp

FFT.o: src/FFT.cpp include/FFT.h
	${CC} ${CFLAGS} -I include -c src/FFT.cpp

pRFDecisionTree.o: include/pRFDecisionTree.h src/pRFDecisionTree.cpp
	${CC} ${CFLAGS} -I include -c src/pRFDecisionTree.cpp

pRandomForest.o: include/pRandomForest.h src/pRandomForest.cpp pRFDecisionTree.o FFT.o
	${CC} ${CFLAGS} -I include -c src/pRandomForest.cpp

clean:
	rm -rf *.o bin/*
