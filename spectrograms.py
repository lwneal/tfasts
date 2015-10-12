import timeit
import random
import os
import struct
import math
import wave

import numpy
import scipy
from PIL import Image
from scipy import signal
from scipy.io import wavfile


KERNEL_WIDTH=16
KERNEL_HEIGHT=16
SPECTROGRAM_HEIGHT = 256


# Requires: Mono 16-bit uncompressed little-endian PCM WAV
def load_wav(filename):
    rate, data = wavfile.read(filename)
    return list(data)


def make_spectrogram(samples):
    f, t, spec = signal.spectrogram(samples, nperseg=SPECTROGRAM_HEIGHT * 2, noverlap=SPECTROGRAM_HEIGHT/2, window='hamming')

    # Remove zeroth element
    data = spec[1:]

    # Zero out low frequencies
    data[:8] = 0

    # Apply filtering
    data = whitening_filter(data)
    # Normalize the spectrogram
    data *= 1.0 / data.max()
    return data


def whitening_filter(spec, sample_pc=0.20):
    # Find the quietest 20% of frames
    height, width = spec.shape
    loudness = [sum(col) for col in spec.transpose()]
    cols = sorted(zip(loudness, range(width)))
    cutoff_idx = int(width * sample_pc)
    _, noise_idxes = zip(*cols[:cutoff_idx])

    # Average them and call it a 'noise profile'
    noise_profile = numpy.ones(height)
    for idx in noise_idxes:
        noise_profile += spec[:,idx]
    noise_profile *= (1.0 / width)

    # Divide each column by its average noise
    for col in range(width):
        for row in range(height):
            spec[row, col] /= noise_profile[row]

    # Heuristic: Also sqrt the spectrogram for visibility
    spec = numpy.power(spec, 0.5)
    return spec


def load_image(filename):
    img = Image.open(filename)
    data = numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0])
    return data


def extract_example(audio_filepath, label_filepath, width=KERNEL_WIDTH, height=KERNEL_HEIGHT):
    # Takes <1ms (thanks, scipy!)
    label = load_image(label_filepath)

    # Takes <1ms (thanks, scipy!)
    wav = load_wav(audio_filepath)

    # Takes 100ms
    spec = make_spectrogram(wav)

    # Takes <1ms
    labels = scipy.misc.imresize(label.transpose(), (spec.shape[0], spec.shape[1])) * (1.0 / 256)
    print "Loaded {} spectrogram size {} with labels min {} max {}".format(
        audio_filepath, spec.shape, labels.min(), labels.max())

    # Takes 200ms
    x_list = []
    y_list = []
    for i in range(height/2, spec.shape[0] - height/2 - 1):
      for j in range(width/2, spec.shape[1] - width/2 - 1):
        label = labels[i, j]
        # HACK: Sample negative examples to keep the size down
        if label == 0 and random.random() < 0.95:
          continue
        x_list.append(extract_feature_vector(spec, i, j))
        y_list.append(label)

    x_vector = numpy.array(x_list)
    y_vector = numpy.array(y_list)
    return x_vector, y_vector


def extract_feature_vector(spec, row, col):
    top, bottom = row - KERNEL_HEIGHT/2, row + KERNEL_HEIGHT/2
    left, right = col - KERNEL_WIDTH/2, col + KERNEL_WIDTH/2
    data = spec[top:bottom, left:right].flatten()
    return data


def extract_examples(audio_dir, label_dir, file_count=None):
    # For each label file, load it and the corresponding audio
    label_filenames = [f for f in os.listdir(label_dir) if f.endswith('.bmp')]
    examples = []
    labels = []
    if file_count:
        random.shuffle(label_filenames)
        label_filenames = label_filenames[:file_count]
    for filename in label_filenames:
        label_filepath = os.path.join(label_dir, filename)
        audio_filepath = os.path.join(audio_dir, filename.replace('bmp', 'wav'))
        x, y = extract_example(audio_filepath, label_filepath)
        if y.min() < 0.0 or y.max() > 1.0:
          print "bad labels out of valid range"
          import pdb; pdb.set_trace()
        examples.extend(x)
        labels.extend(y)
    # Numpy some arrays around?
    examples = numpy.array(examples)
    labels = numpy.array(labels).reshape(len(labels), 1)
    print "Examples mean {0} max {1}".format(numpy.mean(examples), numpy.max(examples))
    print "Labels mean {0} max {1}".format(numpy.mean(labels), numpy.max(labels))
    return examples, labels
