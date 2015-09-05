"""
    Usage:
        learn.py <mnist_training> <mnist_labels> [--epochs=N]

    Arguments:
        -e, --epochs=N       Number of epochs to run [default: 10]


This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import docopt
import numpy
import os
import random
import struct
import sys
import theano
import theano.tensor as T
import time
import timeit
from logistic_sgd import LogisticRegression
from logistic_sgd import sgd_optimization_mnist
from multilayer_perceptron import HiddenLayer
from multilayer_perceptron import test_mlp
from convnet import LeNetConvPoolLayer


BATCH_SIZE=500


def evaluate_lenet5(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y,
                    learning_rate=0.1, n_epochs=10,
                    nkerns=[20, 50], batch_size=BATCH_SIZE):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )


    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    predict_function = theano.function(
            name="Simplified LeNet5",
            inputs=[x],
            outputs=layer3.y_pred)
    return predict_function


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_data(data_filename, label_filename):

    training_data = load_mnist_data(data_filename)
    training_labels = load_mnist_labels(label_filename)

    test_set = (training_data[0:10000], training_labels[0:10000])
    valid_set = (training_data[10000:20000], training_labels[10000:20000])
    train_set = (training_data[20000:60000], training_labels[20000:60000])

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


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
    digits = numpy.zeros((num_items, num_rows * num_cols))
    for i in range(num_items):
        offset = 16 + i * size
        # Normalize MNIST pixels to the range [0,1]
        digits[i] = (1.0 / 256) * numpy.fromstring(data[offset:offset+size], dtype=numpy.uint8)
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
    logistic_labels = numpy.zeros((len(labels), 10),)
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
    epochs = int(arguments['--epochs'])
    training, validation, testing = load_data(arguments['<mnist_training>'], arguments['<mnist_labels>'])

    sgd_classifier = sgd_optimization_mnist(training[0], training[1], validation[0], validation[1], testing[0], testing[1], n_epochs=epochs)
    mlp_classifier = test_mlp(training[0], training[1], validation[0], validation[1], testing[0], testing[1], n_epochs=epochs)
    lenet_classifier = evaluate_lenet5(training[0], training[1], validation[0], validation[1], testing[0], testing[1], n_epochs=epochs, nkerns=[4,10])

    with open('sgd_classifier.pkl', 'w') as f:
        cPickle.dump(sgd_classifier, f)
    with open('mlp_classifier.pkl', 'w') as f:
        cPickle.dump(mlp_classifier, f)
    with open('lenet_classifier.pkl', 'w') as f:
        cPickle.dump(lenet_classifier, f)

    digits = load_mnist_data(arguments['<mnist_training>'])
    random.shuffle(digits)
    for digit in digits:
        sgd_label = sgd_classifier([digit])
        mlp_label = mlp_classifier([digit])
        def underp_lenet(digit):
            derp = numpy.concatenate([digit] * BATCH_SIZE)
            return [lenet_classifier([derp])[0]]
        lenet_label = underp_lenet(digit)
	if sgd_label == mlp_label == lenet_label:
	  continue

        print "Logistic regression predicts {0}, MLP predicts {1}, lenet predicts {2}".format(sgd_label, mlp_label, lenet_label)
        print_digit(digit)
        print "Press enter to continue..."
        print
        raw_input()
