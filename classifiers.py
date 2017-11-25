# Written by David McDougall, 2017

import numpy as np
import random
from genetics import Parameters
from sdr import SDR


class SDRC_Parameters(Parameters):
    parameters = ['alpha',]
    def __init__(self, alpha=1/1000):
        self.alpha = alpha

class SDR_Classifier:
    """Maximum Likelyhood classifier for SDRs."""
    def __init__(self, parameters, input_sdr, output_shape, output_type):
        """
        Argument parameters must be an instance of SDRC_Parameters.
        Argument output_type must be one of: 'index', 'bool', 'pdf'
        """
        self.args         = parameters
        self.input_sdr    = SDR(input_sdr)      # EEK! This copies the arguments current value instead of saving a reference to argument.
        self.output_shape = tuple(output_shape)
        self.output_type  = output_type
        assert(self.output_type in ('index', 'bool', 'pdf'))
        # Don't initialize to zero, touch every input+output pair once or twice.
        self.stats = np.random.uniform(0, 5 * self.args.alpha, size=(self.input_sdr.size,)+self.output_shape)
        self.age = 0

    def train(self, input_sdr, out):
        """
        Argument inp is tuple of index arrays, as output from SP's or TP's compute method
        inp = (ndarray of input space dim 0 indexes, ndarray of input space dim 1 indexes, ...)
        """
        self.input_sdr.assign(input_sdr)
        inp = self.input_sdr.flat_index
        alpha = self.args.alpha
        self.stats[inp] *= (1-alpha)   # Decay
        # Update.
        if self.output_type == 'index':
            # try:
            for out_idx in out:
                self.stats[inp, out_idx] += alpha
            # except TypeError:
            #     self.stats[inp + out] += alpha

        if self.output_type == 'bool':
            self.stats[inp, out] += alpha

        if self.output_type == 'pdf':
            updates = (out - self.stats[inp]) * alpha
            self.stats[inp] += updates

        self.age += 1

    def predict(self, input_sdr=None):
        """
        Argument inputs is ndarray of indexes into the input space.
        Returns probability of each catagory in output space.
        """
        self.input_sdr.assign(input_sdr)
        pdf = self.stats[self.input_sdr.flat_index]
        if True:
            # Combine multiple probabilities into single pdf. Product, not
            # summation, to combine probabilities of independant events. The
            # problem with this is if a few unexpected bits turn on it
            # mutliplies the result by zero, and the test dataset is going to
            # have unexpected things in it.  
            return np.product(pdf, axis=0, keepdims=False)
        else:
            # Use summation B/C it works well.
            return np.sum(pdf, axis=0, keepdims=False)

    def __str__(self):
        s = "SDR Classifier alpha %g\n"%self.args.alpha
        s += "\tInput -> Output shapes are", self.input_shape, '->', self.output_shape
        return s


class RandomOutputClassifier:
    """
    This classifier uses the frequency of the trained target outputs to generate
    random predictions.  It is used to get a baseline  performance to compare
    against the SDR_Classifier.
    """
    def __init__(self, output_shape):
        self.output_shape = tuple(output_shape)
        self.stats = np.zeros(self.output_shape)
        self.age = 0

    def train(self, out):
        """
        Argument out is tuple of index arrays, SDR encoded value of target output
                     Or it can be a dense boolean array too.
        """
        if True:
            # Probability density functions
            self.stats += out / np.sum(out)
        else:
            # Index or mask arrays
            self.stats[out] += 1
        self.age += 1

    def predict(self):
        """
        Argument inputs is ndarray of indexes into the input space.
        Returns probability of each catagory in output space.
        """
        return self.stats
        return np.random.random(self.output_shape) < self.stats / self.age

    def __str__(self):
        return "Random Output Classifier, Output shape is %s"%str(self.output_shape)
