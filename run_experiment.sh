#!/bin/sh

rm output/*
rm comparisons/*
rm demos/*.png
THEANO_FLAGS=device=gpu,floatX=float32 python birds.py --wavs=setA/wavs/ --labels=setA/labels --unlabeled=setB/wavs/ --epochs=500 --output=output
