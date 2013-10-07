CC=clang++ -std=c++11 -stdlib=libc++
CFLAGS=-g

.PHONY: all test test_spectrogram test_learn test_filter clean

all: bin/spectrogram bin/learn bin/filter

test: test_spectrogram test_learn test_filter

test_spectrogram: all
	@rm -rf test/*
	@bin/spectrogram -i hja_birds/wavs/PC7_20090704_090000_0040.wav -o test/test_1.bmp \
	&& bin/spectrogram hja_birds/wavs/PC9_20090512_070000_0090.wav test/test_2.bmp \
	&& bin/spectrogram hja_birds/wavs/PC5_20090703_100000_0010.wav test/test_3.bmp -w 1024 -s 128 -p 50\
	&& echo "Passed tests: spectrogram"\
 	|| echo "Failed tests: spectrogram"

test_learn:
	@bin/learn three_files/ -o test/test_model.rf \
	&& echo "Passed tests: learn"\
 	|| echo "Failed tests: learn"

test_filter:
	@bin/filter hja_birds/wavs/PC13_20090512_070000_0060.wav test/filtered.wav -m test/test_model.rf \
	&& echo "Passed tests: filter"\
 	|| echo "Failed tests: filter"

bin/spectrogram: src/spectrogram.cpp Mask.o pRandomForest.o Image.o
	${CC} ${CFLAGS} *.o src/spectrogram.cpp -I include -o bin/spectrogram

bin/learn: src/learn.cpp Mask.o pRandomForest.o Image.o
	${CC} ${CFLAGS} *.o src/learn.cpp -I include -o bin/learn

bin/filter: src/filter.cpp Mask.o pRandomForest.o Image.o
	${CC} ${CFLAGS} *.o src/filter.cpp -I include -o bin/filter

source.o: include/dlib/all/source.cpp
	${CC} ${CFLAGS} -DDLIB_NO_GUI_SUPPORT -I include -c include/dlib/all/source.cpp

Mask.o: include/Mask.h src/Mask.cpp source.o Grid.o include/Features.h include/Utility.h
	${CC} ${CFLAGS} -I include -c src/Mask.cpp

Grid.o: include/Grid.h src/Grid.cpp source.o
	${CC} ${CFLAGS} -I include -c src/Grid.cpp

Image.o: include/Image.h src/Image.cpp Mask.o Grid.o source.o
	${CC} ${CFLAGS} -I include -c src/Image.cpp

FFT.o: src/FFT.cpp include/FFT.h pWavData.o
	${CC} ${CFLAGS} -I include -c src/FFT.cpp

pRFDecisionTree.o: include/pRFDecisionTree.h src/pRFDecisionTree.cpp
	${CC} ${CFLAGS} -I include -c src/pRFDecisionTree.cpp

pRandomForest.o: include/pRandomForest.h src/pRandomForest.cpp pRFDecisionTree.o FFT.o
	${CC} ${CFLAGS} -I include -c src/pRandomForest.cpp

pWavData.o: include/pWavData.h include/wav_in.h include/wav_out.h
	${CC} ${CFLAGS} -I include -c src/wav_in.cpp src/wav_out.cpp

clean:
	rm -rf *.o bin/* test_*.bmp
