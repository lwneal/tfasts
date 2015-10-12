"""
Usage: 
    eval.py [--equal-precision-recall] <output_dir>

Arguments:
    <output_dir>                    Output spectrogram probability masks
    --equal-precision-recall        If set, output the point where the precision and recall are equal
"""
import docopt
import sys
import os

from PIL import Image
import numpy
import scipy.misc

# Per-pixel metric
def count_labels(label, prediction, threshold):
    assert label.shape == prediction.shape

    tp = len(label[numpy.where((label > 0) & (prediction > threshold))])
    fp = len(label[numpy.where((label == 0) & (prediction > threshold))])
    tn = len(label[numpy.where((label == 0) & (prediction <= threshold))])
    fn = len(label[numpy.where((label > 0) & (prediction <= threshold))])
    return tp, fp, tn, fn


def evaluate(output_dir, label_files, threshold):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for label_filename in label_files:
        label_image = Image.open(label_filename)
        label_data = numpy.asarray(label_image)
        png_filename = label_filename.split('/')[-1].replace('.bmp', '.wav.png')
        prediction_filename = os.path.join(output_dir, png_filename)
        if not os.path.isfile(prediction_filename):
            #print("No label file {0}, skipping".format(prediction_filename))
            continue
        prediction_image = Image.open(prediction_filename)
        prediction_data = numpy.asarray(prediction_image)[:,:,0]
        prediction = scipy.misc.imresize(prediction_data, label_data.shape)
        tp, fp, tn, fn = count_labels(label_data, prediction, threshold)
        true_pos += tp
        false_pos += fp
        true_neg += tn
        false_neg += fn
        recall = 1.0 * true_pos / (0.1 + true_pos + false_neg)
        precision = 1.0 * true_neg / (0.1 + true_neg + false_pos)
    return recall, precision


def ls(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir)]


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    output_dir = arguments['<output_dir>']
    label_files = ls('setA/labels') + ls('setB/labels')
    if arguments['--equal-precision-recall']:
        min_threshold = 0
        max_threshold = 255
        while True:
            threshold = (min_threshold + max_threshold) / 2
            recall, precision = evaluate(output_dir, label_files, threshold)
            print "Threshold {0}\tRecall {1}\tPrecision {2}".format(threshold, recall, precision)
            if recall < precision:
                max_threshold = threshold
            else:
                min_threshold = threshold
            if abs(max_threshold - min_threshold) < 1:
                break
    else:
        for threshold in numpy.arange(0, 255, 5.0):
            recall, precision = evaluate(output_dir, label_files, threshold)
            print "Threshold {0}\tRecall {1}\tPrecision {2}".format(threshold, recall, precision)
