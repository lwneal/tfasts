#!/bin/sh

if [[ $# -lt 4 ]]
then
	echo "Usage: ./reproduce_icassp.sh path/to/training/wavs/ path/to/hja_birds/labels path/to/testing/wavs/  output/path/"
	echo
	echo "For 2-fold validation:"
	echo "./reproduce_icassp.sh setA/wavs/ setA/labels/ setB/wavs/ output/setB/"
	echo "./reproduce_icassp.sh setB/wavs/ setB/labels/ setA/wavs/ output/setA/"
	exit 1
fi

make

echo "Found `ls $1 | wc -l` wav files and `ls $2 | wc -l` label files: loading..."

# Learn a model from one set of .wav recordings (with labels)
bin/learn -i $1 -l $2 -o icassp_model.rf -n 10 -e 50 -c 0.10

# Apply the learned model to another set of .wav recordings
for file in `ls $3 | sed s/.wav$//`
do
	bin/filter -i $3/${file}.wav -o $4/${file}.filtered.wav -a $4/${file}.bmp -m icassp_model.rf
done
