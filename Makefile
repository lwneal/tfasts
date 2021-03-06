CC=clang++ -std=c++11 -stdlib=libc++
CFLAGS=-g

.PHONY: all test test_spectrogram test_learn test_filter clean

all: bin/spectrogram bin/learn bin/filter

test: all
	@rm -rf test/*
	bin/spectrogram -i demo/1.wav -o test/test_1.bmp \
	&& bin/spectrogram demo/2.wav test/test_2.bmp \
	&& bin/spectrogram demo/3.wav test/test_3.bmp -w 1024 -s 128 -p 50\
	&& bin/filter -i demo_wavs/PC3_20090523_190000_0070.wav -o test/PC3_20090523_190000_0070.filtered.wav -l demo_labels/PC3_20090523_190000_0070.bmp \
	&& bin/learn demo/ -o test/test_model.rf \
	&& bin/learn -i demo_wavs -l demo_labels -o test/big_model.rf -e 50 \
	&& cp demo/4.wav test/before.wav \
	&& bin/filter demo/4.wav test/after.wav -m test/test_model.rf \
	&& echo "Passed tests"\
 	|| echo "Failed tests"

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

FFT.o: wav_in.o wav_out.o src/FFT.cpp include/FFT.h
	${CC} ${CFLAGS} -I include -c src/FFT.cpp

pRFDecisionTree.o: include/pRFDecisionTree.h src/pRFDecisionTree.cpp
	${CC} ${CFLAGS} -I include -c src/pRFDecisionTree.cpp

pRandomForest.o: include/pRandomForest.h src/pRandomForest.cpp pRFDecisionTree.o FFT.o
	${CC} ${CFLAGS} -I include -c src/pRandomForest.cpp

wav_in.o: wav_out.o
wav_out.o: include/pWavData.h include/wav_in.h include/wav_out.h include/wav_def.h include/f_err.h include/f_ptch.h src/wav_in.cpp src/wav_out.cpp
	${CC} ${CFLAGS} -I include -c src/wav_in.cpp src/wav_out.cpp

clean:
	rm -rf *.o bin/* test/*
