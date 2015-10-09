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
    output_dir = sys.argv[1]
    label_files = ls('setA/labels') + ls('setB/labels')
    for threshold in numpy.arange(0, 256, 10.0):
        recall, precision = evaluate(output_dir, label_files, threshold)
        print "Threshold {0}\tRecall {1}\tPrecision {2}".format(threshold, recall, precision)
