import os
import struct
import math
import wave

import numpy
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
    data *= 255.0 / data.max()
    return data


def whitening_filter(spec, sample_pc=0.20):
    # Find the quietest 20% of frames
    height, width = spec.shape
    loudness = [sum(col) for col in spec.transpose()]
    cols = sorted(zip(loudness, range(width)))
    cutoff_idx = int(width * sample_pc)
    _, noise_idxes = zip(*cols[:cutoff_idx])

    # Average them and call it a 'noise profile'
    noise_profile = numpy.zeros(height)
    for idx in noise_idxes:
        noise_profile += spec[:,idx]
    noise_profile *= (1.0 / width)

    # Subtract the noise profile from all columns
    for col in range(width):
        for row in range(height):
            spec[row, col] = max(.0, spec[row, col] - noise_profile[row])
    return spec


def load_image(filename):
    img = Image.open(filename)
    return numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0])


PADDING = 8
def extract_example(audio_filepath, label_filepath):
    label = load_image(label_filepath)
    spec = make_spectrogram(load_wav(audio_filepath))
    x = []
    y = []
    for spec_idx in range(PADDING, spec.shape[1] - PADDING):
        label_idx = 1.0 * label.shape[1] / spec.shape[1]
        x.append(spec[:, spec_idx])
        y.append(label[:, label_idx])
    return x, y


def extract_examples(audio_dir, label_dir):
    # For each label file, load it and the corresponding audio
    label_filenames = [f for f in os.listdir(label_dir) if f.endswith('.bmp')]
    examples = []
    labels = []
    for filename in label_filenames[:30]:
        label_filepath = os.path.join(label_dir, filename)
        audio_filepath = os.path.join(audio_dir, filename.replace('bmp', 'wav'))
        x, y = extract_example(audio_filepath, label_filepath)
        examples.extend(x)
        labels.extend(y)
    # Numpy some arrays around?
    examples = numpy.array(examples)
    return examples, labels
