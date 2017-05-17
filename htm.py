# Written by David McDougall, 2017

# High score: 93%
# trained 5 x len(dataset)


import numpy as np
import math
import scipy.ndimage
import scipy.ndimage.interpolation
import random


class RandomDistributedScalarEncoder:
    """
    https://arxiv.org/pdf/1602.05925.pdf
    """
    def __init__(self, resolution, size, on_bits):
        self.resolution = resolution
        self.size = size
        self.on_bits = int(round(on_bits))
        self.output_shape = (size,)

    def encode(self, value):
        # This must be integer division! Everything under the resolution must be removed.
        index = value // self.resolution
        code = np.zeros(self.size, dtype=np.bool)
        for offset in range(self.on_bits):
            h = hash(index + offset)
            code[h % self.size] = True
        return code


class ImageEncoder:
    def __init__(self, input_space):
        self.output_shape = tuple(input_space) + (2,)

    def encode(self, image):
        mean = np.mean(image)
        on_bits  = image >= mean
        off_bits = np.logical_not(on_bits)
        return np.dstack([on_bits, off_bits])

        # Grey level feature
        # I don't think these are useful for MNIST
        on_bits = image >= min(2*mean, 255)
        grey_bits = np.logical_and(image >= mean/2, np.logical_not(on_bits))
        off_bits = np.logical_and(np.logical_not(on_bits), np.logical_not(grey_bits))
        return np.dstack([on_bits, grey_bits, off_bits])


class SpatialPooler:
    """
    This class handles the mini-column structures and the feed forward 
        proximal inputs to each cortical mini-column.


    This implementation is based on but differs from the one described by
    Numenta's Spacial Pooler white paper, (Cui, Ahmad, Hawkins, 2017, "The HTM
    Spacial Pooler - a neocortical...") in two main ways, the boosting function
    and the local inhibition mechanism.


    Logarithmic Boosting Function:
    Numenta uses an exponential boosting function.  See figure 1D, a plot of
    their boosting function.  Notice that the curve intercepts the boost-factor
    axis and has an asymptotes along the activation frequency axis.  The
    activation frequency is by definition constrained to the range [0, 1].

    I use the inverse of their function, which intercepts the activation-frequency
    axis and asypmtotically approaches the boost-factors axis.  Then scale the 
    boost factor such that at the desired sparsity it equals 1.0
          boost_function = -log( activation_frequency )
          scale_factor   = 1 / boost_function( target_sparsity )
          boost_factor   = boost_function( activation_frequency ) * scale_factor
          boost_factor   = log( activation_frequency ) / log( target_sparsity )

    This mechanism has the advantage of having no parameters.
    
    Note: This will get jammed if activation_frequencies are initialized to zero.
          Initialize to far below the target activation instead of zero.


    Faster Local Inhibition:
    Numenta activates the top K most excited columns in each area, where K is
    proportional to the sparsity, and the area is a fixed radius from each
    column which is porportional to the radius of the receptive field.

    This activates the top K most excited columns globally, after normalizing
    all columns by their local area mean and standard deviation.  The local area
    is a gaussian window and the standard deviation of the gaussian is
    proportional to the radius of the receptive field.

    In pseudo code:
        mean_normalized = excitement - gaussian_blur( excitement, radius )
        standard_deviation = sqrt( gaussian_blur( mean_normalized ^ 2, radius ) )
        normalized = mean_normalized / standard_deviation
        activate = top_k( normalized, sparsity * number_of_columns )
    """
    def __init__(self, input_dimensions, column_dimensions, radii=None):
        """
        Argument input_dimensions is tuple of input space dimensions
        Argument column_dimensions is tuple of output space dimensions
        Argument radii is tuple of convolutional radii, must be same length as column_dimensions
                 radii units are the input space units
                 radii is optional, if not given assumes no topology

        If column_dimensions is shorter than input_dimensions then the trailing
        input_dimensions are not convolved over, are instead broadcast to all
        columns which are connected via the convolution in the other dimensions.
        """
        self.input_dimensions  = input_dimensions
        self.column_dimensions = column_dimensions
        self.num_inputs  = np.product(input_dimensions)
        self.num_columns = np.product(column_dimensions)
        self.topology = radii is not None
        self.age = 0

        # TODO: subsample_connections loss % should be a parameter.

        # Columns are identified by their index (which is a single flat index).
        # All columns have the same number of inputs (which are identified by a single flat index).
        # proximal_array[column-index][input-index] = value
        # proximal_sources stores index into flattened input.
        if self.topology:
            self.conv_radii = radii
            self.proximal_sources = self.convolution_connections(input_dimensions, column_dimensions, radii)
            self.subsample_connections(.05)     # Randomly remove some of the input conections.
        else:
            self.proximal_sources = self.dense_connections(input_dimensions, column_dimensions)
            self.subsample_connections(.10)

        # proximal_permanences's shape is the same as proximal_sources's
        self.proximal_permanences = np.random.random(self.proximal_sources.shape)

        self.proximal_coincidence_inc = 0.1
        self.proximal_coincidence_dec = 0.02
        coincidence_ratio = self.proximal_coincidence_inc / self.proximal_coincidence_dec
        print('Coincidence Ratio', coincidence_ratio)
        self.proximal_permenances_threshold = 0.5
        self.column_sparsity = 0.02

        if False:
            # Exponential Boosting Stuff, Delete eventually if never used...
            potential_synapses_per_neuron = self.proximal_sources.shape[1]
            self.boost_factor = math.log(potential_synapses_per_neuron)
            self.boost_factor *= 2
            self.boost_factor *= 2
        self.average_activations_alpha = 0.001
        # Initialize the duty cycles to far below the target duty cycle.
        self.average_activations = np.zeros(self.num_columns) + self.column_sparsity / 10000000000

    def convolution_connections(self, input_dimensions, column_dimensions, radii):
        """
        Sets up the sliding window receptive areas for the spatial pooler
        Returns np.ndarray with shape (num_columns, num_inputs)
        """
        assert(len(column_dimensions) == len(radii))

        # Extended column shape to the input shape
        column_dimensions = tuple(column_dimensions) + (1,)*(len(input_dimensions)-len(column_dimensions))

        # Index offsets into the receptive field, neuron is at center
        window_ranges = [slice(-r, r+1) for r in radii]
        # Broadcast over entirety of extra/non-convolutional dimensions.
        window_ranges += [slice(0, inp_extent) for inp_extent in input_dimensions[len(radii):]]
        # window_index[input-dimension][:] = [coordinates of receptive field]
        window_index = np.mgrid[window_ranges]
        window_index = np.array([dim.flatten() for dim in window_index], dtype=np.float32)

        # Find where the columns are in the input.
        # Assume that they are evenly spaced and that the input space wraps around
        column_ranges = [range(0, size) for size in column_dimensions]
        # column_locations[input-dimension][:] = vector of locations in the input space, one for each column.
        column_locations = [dim.flatten() for dim in np.mgrid[column_ranges]]
        column_locations *= np.divide(input_dimensions, column_dimensions).reshape(len(input_dimensions), 1)

        # Apply the window offsets to each column location and record the resulting indecies.
        column_locations = column_locations.reshape(column_locations.shape + (1,))
        window_index = window_index.reshape((window_index.shape[0], 1, window_index.shape[1]))
        # index[neuron-index][:] = [input index vector]
        index = column_locations + window_index     # Broadcasting
        # print('window_index', window_index.shape, 'column_locations', column_locations.shape, 'index', index.shape)
        index = np.array(np.rint(index), dtype=np.int64)

        # Collapse the input dimension of the index into a single index into the flattened input.
        index = np.ravel_multi_index(index, input_dimensions, mode='wrap')

        return index

    def dense_connections(self, input_dimensions, column_dimensions):
        """Connect every input neuron to every column.  Don't forget to subsample the result."""
        # How to make array of source[neuron][input-num] = input-index
        flat_input_space = np.arange(self.num_inputs)
        return np.stack([flat_input_space]*self.num_columns)

    def subsample_connections(self, loss):
        """Randomly severs 'loss' fraction of inputs from every column."""
        num_inputs = int(round(self.proximal_sources.shape[1] * (1-loss)))
        print("num_inputs", num_inputs)
        shuffled = np.random.permutation(self.proximal_sources.T)
        shuffled = shuffled[:num_inputs, :]
        self.proximal_sources = np.sort(shuffled, axis=0).T

    def compute(self, input_sdr, learn=True):
        """
        Returns tuple of column indecies
        """
        synapses = self.proximal_permanences > self.proximal_permenances_threshold
        if input_sdr.shape == self.input_dimensions:
            # DENSE INPUT
            # Argument input_sdr is a dense boolean input.
            # Gather the inputs, mask out disconnected synapses, and sum for activations.
            input_sdr = np.ravel(input_sdr)
            inputs = input_sdr[self.proximal_sources]
            connected_inputs = np.logical_and(synapses, inputs)
            raw_excitment = np.sum(connected_inputs, axis=1)
        else:
            # SPARSE INPUT
            # Argument input_sdr is an index array into the input-space
            inputs_flat      = np.ravel_multi_index(input_sdr, self.input_dimensions)
            connected_inputs = np.in1d(self.proximal_sources, inputs_flat)
            connected_inputs = connected_inputs.reshape(self.proximal_sources.shape)
            connected_inputs = np.logical_and(synapses, connected_inputs)
            raw_excitment    = np.sum(connected_inputs, axis=1)
        self.zz_raw = raw_excitment.reshape(self.column_dimensions)

        # Boosting
        if learn:   # Don't apply boosting during evaluations
            if True:
                # Logarithmic Boosting Function
                boost = np.log2(self.average_activations) / np.log2(self.column_sparsity)
                boost = np.nan_to_num(boost)
                self.zz_boostd = raw_excitment = boost * raw_excitment
            else:
                # Numenta's Exponential Boosting Function
                mean_duty_cycle = np.mean(self.average_activations)     # Should equal the sparsity...
                boost = np.exp(self.boost_factor * (mean_duty_cycle - self.average_activations))
                self.zz_boostd = raw_excitment = boost * raw_excitment
            self.zz_boostd = self.zz_boostd.reshape(self.column_dimensions)

        # Local Inhibition
        if self.topology:
            # Convert the radii from input space to column space
            col_space_dims = len(self.column_dimensions)
            scale_factor = np.divide(self.column_dimensions, self.input_dimensions[:col_space_dims])
            inhibition_radii = np.multiply(self.conv_radii, scale_factor)

            #
            # FINE TUNING
            #
            # inhibition_radii is used as the standard deviation for the
            # gaussian window which defines the local neighborhood of a column.
            # Areas more distant than inhibition_radii supress each other even
            # though they have no shared input.  Areas closer to each other will
            # suppress each other more.  Divide inhibition_radii by 2 so that
            # 95% of the area in the window has shared input and 68% of the
            # inhibition comes from columns within 1/2 of the convolutional
            # radius.
            #
            # In a normal distribution:
            # 68% of area is within one standard deviation
            # 95% of area is within two standard deviations
            #       This what's currently done.
            # 99% of area is within three standard deviations
            #
            inhibition_radii = np.divide(inhibition_radii, 2)

            raw_excitment = raw_excitment.reshape(self.column_dimensions)
            avg_local_excitment = scipy.ndimage.filters.gaussian_filter(raw_excitment, inhibition_radii)
            local_excitment = raw_excitment - avg_local_excitment
            stddev = np.sqrt(scipy.ndimage.filters.gaussian_filter(local_excitment**2, inhibition_radii))
            self.zz_norm = raw_excitment = np.nan_to_num(local_excitment / stddev)
            self.zz_norm = self.zz_norm.reshape(self.column_dimensions)
            raw_excitment = raw_excitment.flatten()

        # Activating the most excited columns.
        #
        # Numenta specify that they prevent columns with a raw excitment of 0 from activating.
        # I take the top K globally, so unless globally all of the inputs are zero this won't be 
        # an issue.
        #
        k = int(round(self.num_columns * self.column_sparsity))
        k = max(k, 1)
        self.active_columns = np.argpartition(-raw_excitment, k-1)[:k]

        if learn:
            self.age += 1
            # Update the exponential rolling average of each columns activation frequency.
            alpha = self.average_activations_alpha
            self.average_activations *= (1 - alpha)                 # Decay with time
            self.average_activations[self.active_columns] += alpha  # Incorperate this sample

            # Update proximal synapse permenances.
            updates = np.choose(inputs[self.active_columns], 
                                [-self.proximal_coincidence_dec, self.proximal_coincidence_inc])
            updates = np.clip(updates + self.proximal_permanences[self.active_columns], 0.0, 1.0)
            self.proximal_permanences[self.active_columns] = updates

        return np.unravel_index(self.active_columns, self.column_dimensions)

    def make_output_dense(self, output=None):
        """
        Returns the output as a dense boolean ndarray.
        If no output is given, uses the most recently computed output.
        """
        if output is None:
            output = np.unravel_index(self.active_columns, self.column_dimensions)
        output = np.array(output)
        dense = np.zeros(self.column_dimensions, dtype=np.bool)
        dense[output] = True
        return dense

    def plot_boost_functions(self, beta = 15):
        # Generate sample points
        dc = np.linspace(0, 1, 10000)
        from matplotlib import pyplot as plt
        fig = plt.figure(1)
        ax = plt.subplot(111)
        log_boost = lambda f: np.log(f) / np.log(self.column_sparsity)
        exp_boost = lambda f: np.exp(beta * (self.column_sparsity - f))
        logs = [log_boost(f) for f in dc]
        exps = [exp_boost(f) for f in dc]
        plt.plot(dc, logs, 'r', dc, exps, 'b')
        plt.title("Boosting Function Comparison \nLogarithmic in Red, Exponential in Blue (beta = %g)"%beta)
        ax.set_xlabel("Activation Frequency")
        ax.set_ylabel("Boost Factor")
        plt.show()

    def entropy(self, diag=True):
        """
        Calculates the entropy of column activations.

        Result is normalized to range [0, 1]
        A value of 1 indicates that all columns are equally and fully utilized.
        """
        p = self.average_activations
        def entropy(p):
            # Binary entroy function
            p_ = (1 - p)
            s = -p*np.log2(p) -p_*np.log2(p_)
            return np.mean(np.nan_to_num(s))
        e = entropy(p) / entropy(self.column_sparsity)
        if diag:
            print("Inst. SP Entropy %g"%e)
        return e

    def stability(self, inputs, *args):
        """
        Measures the long term stability of the given inputs.

        The first time you call this, pass it a list of inputs.
        It is an error to pass this inputs after the first call. (Append, Replace, or RaiseException)

        Subsequent calls will compare the current stability of the initial inputs against prior calls.
        """
        assert(False)

    def noise_perturbation(self, inp, flip_bits, diag=False):
        """
        Measure the change in SDR overlap after moving some of the ON bits.
        """
        tru = self.compute(inp, learn=False)

        # Move some of the on bits around.
        on_bits  = list(zip(*np.nonzero(inp)))
        off_bits = list(zip(*np.nonzero(np.logical_not(inp))))
        flip_bits = min(flip_bits, min(len(on_bits), len(off_bits)) )
        flip_off = random.sample(on_bits, flip_bits)
        flip_on = random.sample(off_bits, flip_bits)
        noisy = np.array(inp, dtype=np.bool)      # Force copy
        noisy[list(zip(*flip_off))] = False
        noisy[list(zip(*flip_on))] = True

        # Calculate the overlap in SP output after adding noise.
        near = self.compute(noisy, learn=False)
        tru  = set(zip(*tru))
        near = set(zip(*near))
        overlap = len(tru.intersection(near))
        overlap_pct = overlap / len(tru)
        if diag:
            print("SP Noise Robustness (%d flipped) %g"%(flip_bits, overlap_pct))
        return overlap_pct

    def noise_robustness(self, inps, diag=False):
        """
        Plot the noise robustness as a function.
        Argument 'inps' is list of encoded inputs.
        """
        if False:
            # Range         Num Samples     Resolution
            # [0, 10)       20              .5
            # [10, 50)      40              1
            # [50, 100]     11              5
            noises = list(np.arange(20) / 2) + list(np.arange(10, 40)) + list(np.arange(11) * 5 + 50)
        elif False:
            # Exponential progression of noises, samples many orders of magnitude of noise.
            num_samples = 50
            x = np.exp(np.arange(num_samples))
            noises = list(x * 100 / np.max(x))
        else:
            # Number of ON bits in encoded input-space +1
            nz = int(round(np.mean([np.count_nonzero(s) for s in inps[:10]])))
            noises = list(np.arange(nz + 1))
            cutoff = len(noises) // 20          # First 'cutoff' many samples have full accuracy.
            while len(noises) > 100 + cutoff:   # Decimate to a sane number of sample points
                noises = noises[:cutoff] + noises[cutoff::2]

        pct_over = []
        for n in noises:
            z = 0
            for inp in inps:
                z += self.noise_perturbation(inp, n, diag=False)
            pct_over.append(z/len(inps))

        if diag:
            from matplotlib import pyplot as plt
            plt.figure(1)
            plt.plot(noises, pct_over)
            plt.show()

        return noises, pct_over


class SDR_Classifier:
    """
    Maximum Likelyhood classifier for SDRs.
    """
    def __init__(self, input_shape, output_shape, output_sparsity, diag=True):
        self.alpha = 1/1000
        self.input_shape = list(input_shape)
        self.output_shape = list(output_shape)
        self.stats = np.zeros(self.input_shape + self.output_shape)
        self.data_points = 0
        if diag:
            print("SDR Classifier alpha %g"%self.alpha)

    def train(self, inp, out):
        """
        Argument inp is tuple of index arrays, as output from the SP.compute method
        Argument out is tuple of index arrays, SDR encoded value of target output
        inp = (ndarray of input space dim 0 indexes, ndarray of input space dim 1 indexes, ...)
        out = (ndarray of output space dim 0 indexes, ndarray of output space dim 1 indexes, ...)
        """
        self.stats[inp] *= (1-self.alpha)
        self.stats[inp + out] += self.alpha
        self.data_points += 1

    def predict(self, inp):
        """
        Argument inputs is ndarray of indexes into the input space.
        Returns tuple of indecies in output space
        """
        return np.product(self.stats[inp], axis=0)
        return np.sum(self.stats[inp], axis=0)


class KNN_Classifier:
    """
    K-Nearest Neighbors classifier for SDRs.

    This takes too long to run.  Too many dimensions...
    """
    def __init__(self, input_shape, output_shape, k=10, diag=True):
        self.input_shape  = list(input_shape)
        self.output_shape = list(output_shape)
        self.k = k
        self.inputs  = []
        self.outputs = []
        self.kdtree  = None
        if diag:
            print("SDR Classifier K =", k)

    def train(self, inp, out):
        """
        Argument inp is tuple of index arrays, as output from the SP.compute method
        Argument out is tuple of index arrays, SDR encoded value of target output
        inp = (ndarray of input space dim 0 indexes, ndarray of input space dim 1 indexes, ...)
        out = (ndarray of output space dim 0 indexes, ndarray of output space dim 1 indexes, ...)
        """
        # Any existing KD tree will need to be rebuilt to include this new data.
        self.kdtree = None
        # Convert from index arrays to data dense boolean arrays
        dense_inp = np.zeros(self.input_shape, dtype=np.bool)
        dense_inp[inp] = True
        self.inputs.append(dense_inp.reshape(-1))   # reshape(-1) flattens array
        self.outputs.append(out)

    def predict(self, inp):
        """
        Argument inputs is ndarray of indexes into the input space.
        Returns tuple of indecies in output space
        """
        if self.kdtree is None:
            # Build the KD tree
            inputs  = np.stack(self.inputs)
            import scipy.spatial
            # Default leafsize is 10.
            self.kdtree = scipy.spatial.cKDTree(inputs, leafsize=1)

        dense_inp = np.zeros(self.input_shape, dtype=np.bool)
        dense_inp[inp] = True
        dist, pred = self.kdtree.query([dense_inp.flatten()], k=self.k, p=1)
        overlap = np.count_nonzero(inp) - dist/2
        result = np.sum(pred * overlap, axis=0)
        return result


class TemporalPooler():
    """
    """
    def __init__(self, column_dimensions, neurons_per_column):
        self.column_dimensions  = tuple(column_dimensions)
        self.neurons_per_column = neurons_per_column
        self.num_columns = np.product(column_dimensions)
        self.num_neurons = neurons_per_column * self.num_columns

        # basal_sources[neuron-index][input-number] = input-index
        sources = np.arange(self.num_neurons)
        self.basal_sources = np.stack([sources]*self.num_neurons)
        self.subsample_connections(0.75)
        self.basal_permanences = np.random.random(self.basal_sources.shape)

        self.basal_permanence_inc = 0.10
        self.basal_permanence_dec = 0.02
        self.basal_permenances_threshold = 0.5

        self.reset()

    def subsample_connections(self, loss):
        """Randomly severs 'loss' fraction of inputs from every column."""
        num_inputs = int(round(self.basal_sources.shape[1] * (1-loss)))
        print("num_inputs", num_inputs)
        shuffled = np.random.permutation(self.basal_sources.T)
        shuffled = shuffled[:num_inputs, :]
        self.basal_sources = np.sort(shuffled, axis=0).T

    def reset(self):
        self.state = np.zeros(self.num_neurons)

    def compute(self, column_activations):
        """
        Returns index array of active neurons.
        """
        # Flatten everything
        columns = np.ravel_multi_index(column_activations, self.column_dimensions)

        # Active columns accumulate input
        # source_index[active-neuron][num-inputs] = source-neuron-index
        source_index = self.basal_sources[columns]
        excitement = np.sum(self.state[source_index], axis=1)

        # Activate Neurons
        pass
        self.anomaly = None

        # Learn
        pass



