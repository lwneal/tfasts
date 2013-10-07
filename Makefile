CC=clang++ -std=c++11 -stdlib=libc++
CFLAGS=-g

.PHONY: all test test_spectrogram test_learn clean

all: bin/spectrogram bin/learn

test: test_spectrogram test_learn

test_spectrogram: all
	@rm -rf test/*
	@bin/spectrogram -i hja_birds/wavs/PC7_20090704_090000_0040.wav -o test/test_1.bmp \
	&& bin/spectrogram hja_birds/wavs/PC9_20090512_070000_0090.wav test/test_2.bmp \
	&& bin/spectrogram hja_birds/wavs/PC5_20090703_100000_0010.wav test/test_3.bmp -w 1024 -s 128 -p 50\
	&& echo "Passed tests: spectrogram"\
 	|| echo "Failed tests: spectrogram"
	open test/*.bmp

test_learn: all
	@rm -rf test/*
	@bin/learn three_files/ -o test/test_model.rf \
	&& echo "Passed tests: learn"\
 	|| echo "Failed tests: learn"

bin/spectrogram: src/spectrogram.cpp Mask.o pRandomForest.o Image.o
	${CC} ${CFLAGS} *.o src/spectrogram.cpp -I include -o bin/spectrogram

bin/learn: src/learn.cpp Mask.o pRandomForest.o Image.o
	${CC} ${CFLAGS} *.o src/learn.cpp -I include -o bin/learn

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
