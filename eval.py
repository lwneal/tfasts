import os

from PIL import Image
import numpy

def count_labels(label, prediction, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    height_factor = 1.0 * label.shape[0] / prediction.shape[0]
    width_factor = 1.0 * label.shape[1] / prediction.shape[1]
    for row in range(len(label)):
        for col in range(len(label)):
            l = label[row, col]
            p = prediction[int(row * height_factor), int(col * width_factor)]
            if l > threshold and p > threshold:
                tp += 1
            elif l > threshold and not p > threshold:
                fn += 1
            elif not l > threshold and p > threshold:
                fp += 1
            elif not l > threshold and not p > threshold:
                tn += 1
    #print "tp/fp/tn/fn: {0} {1} {2} {3}".format(tp, fp, tn, fn)
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
            print("No label file {0}, skipping".format(prediction_filename))
            continue
        prediction_image = Image.open(prediction_filename)
        prediction_data = numpy.asarray(prediction_image)[:,:,0]
        tp, fp, tn, fn = count_labels(label_data, prediction_data, threshold)
        true_pos += tp
        false_pos += fp
        true_neg += tn
        false_neg += fn
        recall = 1.0 * true_pos / (0.1 + true_pos + false_neg)
        precision = 1.0 * true_neg / (0.1 + true_neg + false_pos)
        print "Recall {} precision {}".format(recall, precision)
    print "Total:"
    print "True positive: {0}".format(true_pos)
    print "False positive: {0}".format(false_pos)
    print "True negative: {0}".format(true_neg)
    print "False negative: {0}".format(false_neg)
    return recall, precision


def ls(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir)]

if __name__ == '__main__':
    output_dir = 'output/'
    label_files = ls('setA/labels') + ls('setB/labels')
    for i in range(10):
        threshold =  255.0 * (i) / 20.0
        recall, precision = evaluate(output_dir, label_files, threshold * 255.0)
        print "Threshold {0}\tRecall {1}\tPrecision {2}".format(threshold, recall, precision)
