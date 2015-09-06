"""
Usage:
  birds.py [<audio_dir> <label_dir>] [--unlabeled_audio_dir=DIR]

Options:
  <audio_dir>       Directory containing .wav audio files
  <label_dir>       Directory containing .bmp binary spectrogram masks
  DIR               Directory containing .wav audio files that are not labeled

Ensure .wav files are 16-bit mono PCM at 16khz.
Ensure .bmp have 256 pixels height
"""
import os
import docopt
import random
import cPickle

import numpy
from PIL import Image
from learn import shared_dataset
import spectrograms
from spectrograms import extract_examples, load_wav, make_spectrogram
from spectrograms import PADDING

from multilayer_perceptron import test_mlp


def load_data(audio_dir, label_dir):
    training_data, training_labels = extract_examples(audio_dir, label_dir)
    data = zip(training_data, training_labels)
    random.shuffle(data)
    training_data, training_labels = zip(*data)

    # TODO: Learn to output a vector
    training_labels = [1 if any(l) else 0 for l in training_labels]
    assert len(training_data) == len(training_labels)

    idx = len(training_data) / 3
    test_set = (training_data[:idx], training_labels[:idx])
    valid_set = (training_data[idx:2*idx], training_labels[idx:2*idx])
    train_set = (training_data[2*idx:], training_labels[2*idx:])
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def demonstrate_classifier(audio_dir, classifier):
    for filename in os.listdir(audio_dir):
        samples = load_wav(os.path.join(audio_dir, filename))
        spec = make_spectrogram(samples)
        label = numpy.zeros(spec.shape)
        for col in range(PADDING, spec.shape[1] - PADDING):
            input_x = spec[:, col]
            output_y = classifier([input_x])[0]
            if output_y == 0:
                label[:, col] = 0
            else:
                label[:, col] = 255
        comparison = numpy.concatenate( [spec, label] )
        Image.fromarray(comparison).show()


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    if arguments['<audio_dir>']:
        audio_dir = os.path.expanduser(arguments['<audio_dir>'])
        label_dir = os.path.expanduser(arguments['<label_dir>'])
        training, validation, testing = load_data(audio_dir, label_dir)
        mlp_classifier = test_mlp(training[0], training[1],
            validation[0], validation[1],
            testing[0], testing[1],
            n_epochs=100, n_in=256, n_hidden=3000)
        with open('birds_mlp_classifier.pkl', 'w') as f:
            cPickle.dump(mlp_classifier, f)
    else:
        with open('birds_mlp_classifier.pkl') as f:
            mlp_classifier = cPickle.load(f)

    if arguments['--unlabeled_audio_dir']:
        demonstrate_classifier(arguments['--unlabeled_audio_dir'], mlp_classifier)
