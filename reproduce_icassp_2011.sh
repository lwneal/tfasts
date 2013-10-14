#!/bin/sh

if [[ $# -lt 3 ]]
then
	echo "Usage: ./reproduce_icassp.sh path/to/hja_birds/wavs/ path/to/hja_birds/labels output/path/"
	exit 1
fi

echo "Found `ls $1 | wc -l` wav files and `ls $2 | wc -l` label files: loading..."

if [[ -n icassp_model.rf ]]
then
	# Learn a model with the same parameters used for the 2011 ICASSP paper
	bin/learn -i $1 -l $2 -o icassp_model.rf -n 10 -e 50 -c 0.10
fi

# Apply the learned model
for file in `ls $1 | sed s/.wav$//`
do
	bin/filter -i $1/${file}.wav -o ${file}.filtered.bmp -a ${file}.bmp
done
