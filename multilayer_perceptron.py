"""
    Usage:
        learn.py <mnist_training> <mnist_labels>
"""
import struct
import cPickle
import gzip
import os
import sys
import timeit
import random
import time

import docopt
import numpy
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

from logistic_sgd import LogisticRegression
from convnet import LeNetConvPoolLayer


# early-stopping parameters
patience_increase = 10  # wait this much longer when a new best is found
improvement_threshold = 0.9998  # An improvement of less than this is ignored


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        def relu(x):
          return T.switch(x < 0, 0, x)

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=relu
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input

        # pass through for results of output layer
        self.y_pred = self.logRegressionLayer.y_pred


def test_mlp(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y,
             learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             batch_size=20, n_hidden=500, n_in=32*32, n_out=2):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    nkerns = [13, 17]

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    y = T.matrix('y')  # Labels are masks over the spectrogram

    rng = numpy.random.RandomState(1234)

    layer0_input = x.reshape((batch_size, 1, 32, 32))

    # input: 32*32
    # filtered: (32-5+1) = 28*28
    # pooled: 28/2 = 14*14
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 14, 14)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 32, 32),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2,2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (14-3+1, 14-3+1) = (12, 12)
    # maxpooling reduces this further to (12/2, 12/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 14, 14),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2,2)
    )

    # Output of layer1 is (batch_size, nkerns[1] * 6 * 6)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=n_hidden,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=n_out)

    # TODO: Add regularization
    cost = (
        layer3.negative_log_likelihood(y)
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    start_time = timeit.default_timer()

    best_validation_loss, best_iter, test_score = train_models(
            train_model, validate_model, test_model,
            n_epochs, n_train_batches, n_valid_batches, n_test_batches)

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    predict_model = theano.function(
            inputs=[layer0_input],
            outputs=layer3.y_pred)
    return predict_model


def train_models(
        train_model, validate_model, test_model, 
        n_epochs, n_train_batches, n_valid_batches, n_test_batches):
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    patience = 20000  # look as this many examples regardless
    epoch = 0
    while epoch < n_epochs:
        try:
            best_validation_loss, best_iter, test_score, patience, finished_early = learn_epoch(
                train_model, validate_model, test_model,
                epoch, n_train_batches, n_valid_batches, n_test_batches,
                best_validation_loss, best_iter, test_score, patience)
            epoch += 1
        except KeyboardInterrupt as e:
            print e
            finished_early = True
        if finished_early: 
            print "Ran out of patience, finishing early"
            break
    print "Finished"
    return best_validation_loss, best_iter, test_score


def learn_epoch(train_model, validate_model, test_model, 
        epoch, n_train_batches, n_valid_batches, n_test_batches,
        best_validation_loss, best_iter, test_score, patience):
    for minibatch_index in range(n_train_batches):
        best_validation_loss, best_iter, test_score, patience = train_minibatch(
                train_model, validate_model, test_model, 
                epoch, minibatch_index, n_train_batches, n_valid_batches, n_test_batches,
                best_validation_loss, best_iter, test_score, patience)
        if patience <= calculate_current_iteration(epoch, n_train_batches, minibatch_index):
            return best_validation_loss, best_iter, test_score, patience, True
    return best_validation_loss, best_iter, test_score, patience, False


def calculate_current_iteration(epoch, n_train_batches, minibatch_index):
    # the index of the current iteration
    return (epoch - 1) * n_train_batches + minibatch_index


def train_minibatch(
        train_model, validate_model, test_model, 
        epoch, minibatch_index, n_train_batches, n_valid_batches, n_test_batches,
        best_validation_loss, best_iter, test_score, patience):

    minibatch_avg_cost = train_model(minibatch_index)
    current_iter = calculate_current_iteration(epoch, n_train_batches, minibatch_index)
    validation_frequency = min(n_train_batches, patience / 2)
    if (current_iter + 1) % validation_frequency == 0:
        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i
                             in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        print('epoch %i, minibatch %i/%i, validation error %f ' %
            (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            #improve patience if loss improvement is good enough
            if this_validation_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, current_iter * patience_increase)

            best_validation_loss = this_validation_loss
            best_iter = current_iter

            # test it on the test set
            test_losses = [test_model(i) for i in xrange(n_test_batches)]
            test_score = numpy.mean(test_losses)

            print(('     epoch %i, minibatch %i/%i, test error of '
                   'best model %f ') %
                  (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
    return best_validation_loss, best_iter, test_score, patience
