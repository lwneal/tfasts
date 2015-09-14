"""
Usage:
  birds.py --wavs=DIR --labels=DIR [--unlabeled=DIR] [--file-count=COUNT] [--save-data=NAME] [--load-data=NAME] [--epochs=EPOCHS]

Options:
  --wavs=DIR               Directory containing training .wav files
  --labels=DIR             Directory containing .bmp binary spectrogram masks
  --unlabeled=DIR          Directory containing testing .wav files
  --file-count=COUNT       Max number of inputs to read
  --save-data=NAME         Location to save classifier
  --load-data=NAME         Location to read previously-trained classifier
  --epochs=EPOCHS          Number of epochs to run [default: 10]

Ensure .wav files are 16-bit mono PCM at 16khz.
Ensure .bmp have 256 pixels height
"""
import os
from os.path import expanduser
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
            input_list = []
            for offset in [-3, -2, -1, 0, 1, 2, 3]:
                input_list.append(spec[:, col + offset])
            input_x = numpy.concatenate(input_list, axis=1)
            label[:, col] = classifier([input_x])[0]
        print("Label mean is {0} max is {1}".format(numpy.mean(label), numpy.max(label)))
        comparison = numpy.concatenate( [spec, label] ) * 255.0
        #Image.fromarray(comparison).show()
        img = Image.fromarray(label * 255).convert('RGB')
        img.save('output/' + filename + '.png')


def train_classifier(wav_dir, label_dir, num_epochs, file_count=None):
    training, validation, testing = load_data(wav_dir, label_dir, file_count)
    mlp_classifier = test_mlp(training[0], training[1],
        validation[0], validation[1],
        testing[0], testing[1],
        n_epochs=num_epochs, n_in=7 * 256, n_out=256, n_hidden=256, learning_rate=3.0)
    with open('birds_mlp_classifier.pkl', 'w') as f:
        cPickle.dump(mlp_classifier, f)
    return mlp_classifier


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    num_epochs = int(arguments['--epochs'])
    wav_dir = expanduser(arguments['--wavs'])
    labels = expanduser(arguments['--labels'])
    unlabeled_dir = expanduser(arguments['--unlabeled']) if arguments['--unlabeled'] else None
    file_count = int(arguments['--file-count']) if arguments['--file-count'] else None

    if wav_dir:
        mlp_classifier = train_classifier(wav_dir, labels, num_epochs, file_count)
    else:
        print("Loading classifier from file...")
        with open('birds_mlp_classifier.pkl') as f:
            mlp_classifier = cPickle.load(f)

    if unlabeled_dir:
        demonstrate_classifier(arguments['--unlabeled'], mlp_classifier)
