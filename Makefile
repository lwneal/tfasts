CC=clang++ -std=c++11 -stdlib=libc++
CFLAGS=-g

.PHONY: all test clean dlib

all: spectrogram

test: all
	bin/spectrogram -i hja_birds/wavs/PC7_20090704_090000_0040.wav -o test_1.bmp
	open test_*.bmp

spectrogram: src/spectrogram.cpp Mask.o pRandomForest.o Image.o
	${CC} ${CFLAGS} *.o src/spectrogram.cpp -I include -o bin/spectrogram

dlib: source.o
source.o: include/dlib/all/source.cpp
	${CC} ${CFLAGS} -DDLIB_NO_GUI_SUPPORT -I include -c include/dlib/all/source.cpp

Mask.o: include/Mask.h src/Mask.cpp source.o Grid.o
	${CC} ${CFLAGS} -I include -c src/Mask.cpp

Grid.o: include/Grid.h src/Grid.cpp source.o
	${CC} ${CFLAGS} -I include -c src/Grid.cpp

Image.o: include/Image.h src/Image.cpp Mask.o Grid.o source.o
	${CC} ${CFLAGS} -I include -c src/Image.cpp

FFT.o: src/FFT.cpp include/FFT.h
	${CC} ${CFLAGS} -I include -c src/FFT.cpp

pRFDecisionTree.o: include/pRFDecisionTree.h src/pRFDecisionTree.cpp
	${CC} ${CFLAGS} -I include -c src/pRFDecisionTree.cpp

pRandomForest.o: include/pRandomForest.h src/pRandomForest.cpp pRFDecisionTree.o FFT.o
	${CC} ${CFLAGS} -I include -c src/pRandomForest.cpp

clean:
	rm -rf *.o bin/* test_*.bmp
