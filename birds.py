"""
Usage:
  birds.py [<audio_dir> <label_dir>] [--unlabeled_audio_dir=DIR] [--file-count=COUNT] [--save-data=SAVENAME] [--load-data=LOADNAME]

Options:
  <audio_dir>       Directory containing .wav audio files
  <label_dir>       Directory containing .bmp binary spectrogram masks
  DIR               Directory containing .wav audio files that are not labeled
  COUNT             Maximum number of files to train on
  

Ensure .wav files are 16-bit mono PCM at 16khz.
Ensure .bmp have 256 pixels height
"""
import os
import docopt
import random
import cPickle
import time

import numpy
from PIL import Image
import spectrograms
from spectrograms import extract_examples, load_wav, make_spectrogram
from spectrograms import PADDING
import theano

from multilayer_perceptron import test_mlp


def load_data(audio_dir, label_dir, file_count=None):
    training_data, training_labels = extract_examples(audio_dir, label_dir, file_count)
    data = zip(training_data, training_labels)
    random.shuffle(data)
    training_data, training_labels = zip(*data)

    #training_labels = [1 if any(l) else 0 for l in training_labels]
    assert len(training_data) == len(training_labels)

    idx = len(training_data) / 3
    rval = [(training_data[:idx], training_labels[:idx]), 
            (training_data[idx: 2*idx], training_labels[idx:2*idx]),
            (training_data[2*idx:], training_labels[2*idx:])]

    def shared(data):
      return theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)
    return [(shared(train), shared(test)) for (train, test) in rval]


def demonstrate_classifier(audio_dir, classifier):
    for filename in os.listdir(audio_dir):
        samples = load_wav(os.path.join(audio_dir, filename))
        spec = make_spectrogram(samples)
        label = numpy.zeros(spec.shape)
        for col in range(PADDING, spec.shape[1] - PADDING):
            input_x = spec[:, col]
            label[:, col] = classifier([input_x])[0]
        print("Label mean is {0} max is {1}".format(numpy.mean(label), numpy.max(label)))
        comparison = numpy.concatenate( [spec, label] ) * 255.0
        Image.fromarray(comparison).show()
        time.sleep(1)


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    if arguments['<audio_dir>']:
        audio_dir = os.path.expanduser(arguments['<audio_dir>'])
        label_dir = os.path.expanduser(arguments['<label_dir>'])
        file_count = int(arguments['--file-count']) if arguments['--file-count'] else None
        if arguments['--load-data']:
            fname = arguments['--load-data']
            print("Loading data with name {0}".format(fname))
            with open(fname + 'training.pkl') as f:
                training = cPickle.load(f)
            with open(fname + 'validation.pkl') as f:
                validation = cPickle.load(f)
            with open(fname + 'testing.pkl') as f:
                testing = cPickle.load(f)
            print("Finished loading data")
        else:
            training, validation, testing = load_data(audio_dir, label_dir, file_count)
            if arguments['--save-data']:
                fname = arguments['--save-data']
                print("Saving data with name {0}".format(fname))
                with open(fname + 'training.pkl', 'w') as f:
                    cPickle.dump(training, f)
                with open(fname + 'validation.pkl', 'w') as f:
                    cPickle.dump(validation, f)
                with open(fname + 'testing.pkl', 'w') as f:
                    cPickle.dump(testing, f)
                print("Finished saving data")

        mlp_classifier = test_mlp(training[0], training[1],
            validation[0], validation[1],
            testing[0], testing[1],
            n_epochs=9000, n_in=256, n_out=256, n_hidden=256, learning_rate=.3)
        with open('birds_mlp_classifier.pkl', 'w') as f:
            cPickle.dump(mlp_classifier, f)
    else:
        with open('birds_mlp_classifier.pkl') as f:
            mlp_classifier = cPickle.load(f)

    if arguments['--unlabeled_audio_dir']:
        demonstrate_classifier(arguments['--unlabeled_audio_dir'], mlp_classifier)
