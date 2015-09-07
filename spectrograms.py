import random
import os
import struct
import math
import wave

import numpy
import scipy
from PIL import Image
from scipy import signal


# Requires: Mono 16-bit uncompressed little-endian PCM WAV
def load_wav(filename):
    wavfile = wave.open(filename)
    assert wavfile.getsampwidth() == 2
    assert wavfile.getnchannels() == 1
    samples = []
    for i in range(wavfile.getnframes()):
        sample = struct.unpack('<h', wavfile.readframes(1))
        samples.extend(sample)
    freq = wavfile.getparams()[2]
    len_sec = 1.0 * len(samples) / freq
    print "Loaded audio file length {0} seconds".format(len_sec)
    return samples


def make_spectrogram(samples, desired_height=256):
    spec = signal.spectrogram(samples, nperseg=512, noverlap=128, window='hamming')
    # Zero out low frequencies, remove zeroth element
    spec[2][:8] = 0
    data = spec[2][1:]
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
    for row in range(data.shape[0]):
      for col in range(data.shape[1]):
        data[row][col] = 1.0 if data[row][col] > 0 else 0
    return data


PADDING = 8
def extract_example(audio_filepath, label_filepath):
    label = load_image(label_filepath)
    spec = make_spectrogram(load_wav(audio_filepath))
    x = spec.transpose()
    y = scipy.misc.imresize(label.transpose(), x.shape)
    return x, y


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
    labels = numpy.array(labels)
    print "Examples mean {0} max {1}".format(numpy.mean(examples), numpy.max(examples))
    print "Labels mean {0} max {1}".format(numpy.mean(labels), numpy.max(labels))
    return examples, labels
