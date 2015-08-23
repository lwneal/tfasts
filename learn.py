"""
    Usage:
        learn.py <training_data_file> <test_data_file>
"""
import sys
import docopt
import numpy as np
import theano
import theano.tensor as T
import struct



def logistic_regression(training_vectors, training_labels):
    rng = np.random

    feats = len(training_vectors[0])
    classes = len(training_labels[0])
    training_steps = 10

    # Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.vector("y")
    w = theano.shared(rng.randn(classes), name="w")
    b = theano.shared(np.zeros(feats), name="b")
    print "Initial model:"
    print w.get_value(), b.get_value()

    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))         # Probability that target = 1
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)   # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()      # The cost to minimize
    gw, gb = T.grad(cost, [w, b])                   # Compute the gradient of the cost

    # Compile
    train = theano.function(
            inputs=[x,y],
            outputs=[p_1, xent],
            updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

    predict = theano.function(inputs=[x], outputs=p_1)

    # Train
    for i in range(training_steps):
        import pdb; pdb.set_trace()
        pred, err = train(training_vectors, training_labels)
        print "iteration {0}\terr is {1}".format(i, err)

    print "Final model:"
    print w.get_value(), b.get_value()
    for digit, label in zip(training_vectors, training_labels):
        print_digit(digit)
        print "Predition: {0}\tTrue Label: {1}".format(predict((digit,label)), label)


def load_mnist_data(filename, print_digits=False):
    data = open(filename).read()
    if not data.startswith('\x00\x00\x08\x03'):
        print "Invalid magic number (expected 0x00000803) for MNIST training data file {0}".format(filename)
        exit(1)
    if len(data) < 16:
        print "Invalid size for MNIST training data file"
        exit(1)
    num_items = struct.unpack('>L', data[4:8])[0]
    num_rows = struct.unpack('>L', data[8:12])[0]
    num_cols = struct.unpack('>L', data[12:16])[0]
    size = num_rows * num_cols
    if len(data) != 16 + size * num_items:
        print "Invalid size {0} for MNIST training data file".format(len(data))
        exit(1)
    digits = np.zeros((num_items, num_rows * num_cols))
    for i in range(num_items):
        offset = 16 + i * size
        digits[i] = np.fromstring(data[offset:offset+size], dtype=np.uint8)
    return digits


def load_mnist_labels(filename):
    data = open(filename).read()
    if not data.startswith('\x00\x00\x08\x01'):
        print "Invalid magic number (expected 0x00000801) for MNIST training label file {0}".format(filename)
        exit(1)
    num_items = struct.unpack('>L', data[4:8])[0]
    labels = []
    for value in data[8:]:
        labels.append(ord(value))
    if len(labels) != num_items:
        print "Incorrect number of items {0} (expected {1})".format(len(labels), num_items)
        exit(1)
    return labels


def convert_logistic_label_format(labels):
    logistic_labels = np.zeros((len(labels), 10),)
    for i, label in enumerate(labels):
        logistic_labels[i][label] = 1.0
    return logistic_labels


def print_digit(digit, cols=28):
    for y in range(len(digit) / cols):
        for x in range(cols):
            sys.stdout.write('X' if digit[x + y*cols] else ' ')
        sys.stdout.write('\n')


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    print "Loading mnist data..."
    training_data = load_mnist_data(arguments['<training_data_file>'])
    training_labels = load_mnist_labels(arguments['<test_data_file>'])

    print "Loaded {0} training data examples and {1} labels".format(len(training_data), len(training_labels))
    print "First digit: {0}".format(training_labels[0])
    print_digit(training_data[0])
    print "Last digit: {0}".format(training_labels[-1])
    print_digit(training_data[-1])

    logistic_regression(training_data, convert_logistic_label_format(training_labels))

