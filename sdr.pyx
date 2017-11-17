# Written by David McDougall, 2017
# cython: language_level=3
# cython: profile=True

import numpy as np
cimport numpy as np
cimport cython
import math
from fractions import Fraction
import copy

def _choose(n, r):
    """Number of ways to choose 'r' unique elements from set of 'n' elements."""
    factorial = math.factorial
    return factorial(n) // factorial(r) // factorial(n-r)

def _binary_entropy(p):
    p_ = (1 - p)
    s = -p*np.log2(p) -p_*np.log2(p_)
    return np.mean(np.nan_to_num(s))


class SparseDistributedRepresentation:
    """
    This class represents both the specification and the momentary value of a
    Sparse Distributed Representation.  Classes initialized with SDRs can expect
    the SDRs values to change.

    Attribute dimensions is a tuple of integers, read only.
    Attribute size is the total number of bits in SDR, read only.

    The following three attributes hold the current value of the SDR, and are 
    read/writable.  Values assigned to one attribute are converted to the other
    formats when needed.
    Attribute dense ... ndarray, shape=self.dimensions, dtype=np.int8
    Attribute index ... tuple of ndarray, ndarray.shape=(num-active-bits), dtype=np.int
    Attribute flat_index ... ndarray, shape=(all-one-dim,), dtype=np.int
              The flat index is the index into the flattened SDR.
    """
    def __init__(self, specification,
        activation_frequency_alpha=None,
        average_overlap_alpha=None):
        """
        Argument specification can be either one of the following:
            A tuple of numbers, which declares the dimensions.
            An SDR, which makes this instance a shallow copy of it.

        Optional Argument activation_frequency_alpha.  If given, this SDR will
            automatically track the moving exponential average of each bits
            activation frequency.  The activation frequencies are updated every
            time this SDR is assigned to.  These records are NOT copied to any
            other SDRs, even by the copy constructor.  The given alpha is the
            weight given to each new activation.  Giving a False value
            (including None and 0) will disable this feature entirely, it is OFF
            by default.  The activation frequencies are available at attribute
            self.activation_frequency, dtype=np.float32, shape=(self.size,).

        Optional Argument average_overlap_alpha.  If given, this SDR will
            automatically track the moving exponential average of the overlap
            between sucessive values of this SDR and store them at the attribute
            self.average_overlap.  This is a measurement of how fast the SDR is
            changing.  This value is updated after every assignment to this SDR.
            This argument is NOT copied to any other SDRs, even by the copy
            constructor.  The given alpha is the weight given to each overlap.
            Giving a False value (including None and 0) will disable this
            feature entirely, it is OFF by default.  Attribute
            self.average_overlap contains the output, type=float, range=[0, 1].
        """
        # Private attribute self._callbacks is a list of functions of self, all
        # of which are called each time this SDR's value changes.
        self._callbacks = []
        if isinstance(specification, SDR):
            self.dimensions = specification.dimensions
            self.size       = specification.size
            self.assign(specification)
        else:
            self.dimensions = tuple(int(round(x)) for x in specification)
            self.size       = np.product(self.dimensions)
            self.zero()

        if bool(activation_frequency_alpha):
            self.activation_frequency_alpha = activation_frequency_alpha
            # TODO: Why flat? Dimensions were requested and are expected!
            self.activation_frequency       = np.zeros(self.size, dtype=np.float32)
            self._callbacks.append(type(self)._track_activation_frequency)

        if bool(average_overlap_alpha):
            self.average_overlap_alpha = average_overlap_alpha
            self.average_overlap       = 0.
            self._prev_value           = SDR(self)
            self._callbacks.append(type(self)._track_average_overlap)

    @property
    def dense(self):
        if self._dense is None:
            if self._flat_index is not None:
                self._dense = np.zeros(self.size, dtype=np.int8)
                self._dense[self._flat_index] = 1
                self._dense.shape = self.dimensions
            elif self._index is not None:
                self._dense = np.zeros(self.dimensions, dtype=np.int8)
                self._dense[self._index] = 1
        return self._dense
    @dense.setter
    def dense(self, value):
        assert(isinstance(value, np.ndarray))
        value.shape      = self.dimensions
        value.dtype      = np.int8
        self._dense      = value
        self._index      = None
        self._flat_index = None
        self._handle_callbacks()

    @property
    def index(self):
        if self._index is None:
            if self._flat_index is not None:
                self._index = np.unravel_index(self._flat_index, self.dimensions)
            elif self._dense is not None:
                self._index = np.nonzero(self._dense)
        return self._index
    @index.setter
    def index(self, value):
        value = tuple(value)
        assert(len(value) == len(self.dimensions))
        assert(all(idx.shape == value[0].shape for idx in value))
        self._dense      = None
        self._index      = value
        self._flat_index = None
        self._handle_callbacks()

    @property
    def flat_index(self):
        if self._flat_index is None:
            if self._index is not None:
                self._flat_index = np.ravel_multi_index(self._index, self.dimensions)
            elif self._dense is not None:
                self._flat_index = np.nonzero(self._dense.reshape(-1))[0]
        return self._flat_index
    @flat_index.setter
    def flat_index(self, value):
        assert(len(value.shape) == 1)
        self._flat_index = value
        self._index = None
        self._dense = None
        self._handle_callbacks()

    def assign(self, sdr):
        """
        Accepts an argument of unknown type and assigns it into this SDRs current
        value.  This accepts an SDR instance, dense boolean array, index tuple,
        flat index, or None.  If None is given then this takes no action and
        retains its current value.
        """
        if sdr is None:
            return
        if sdr is self:
            return
        self._dense      = None
        self._index      = None
        self._flat_index = None

        if isinstance(sdr, SDR):
            assert(self.dimensions == sdr.dimensions)
            if sdr._dense is not None:
                self._dense = sdr._dense
            if sdr._index is not None:
                self._index = sdr._index
            if sdr._flat_index is not None:
                self._flat_index = sdr._flat_index
            self._handle_callbacks()

        elif isinstance(sdr, np.ndarray):
            if sdr.dtype in (np.int8, np.uint8, bool) and self.size == np.product(sdr.shape):
                self.dense = sdr
            elif len(sdr.shape) == 1:
                self.flat_index = sdr
        else:
            self.index = sdr

        if self._dense is None and self._index is None and self._flat_index is None:
            raise TypeError("Could not assign %s into an SDR."%type(sdr).__name__)

    def _handle_callbacks(self):
        for func in self._callbacks:
            func(self)

    def _track_activation_frequency(self):
        alpha = self.activation_frequency_alpha
        self.activation_frequency *= (1 - alpha)             # Decay with time
        self.activation_frequency[self.flat_index] += alpha  # Incorperate this sample

    def _track_average_overlap(self):
        alpha                = self.average_overlap_alpha
        overlap              = self.overlap(self._prev_value)
        self.average_overlap = (1 - alpha) * self.average_overlap + alpha * overlap
        self._prev_value     = SDR(self)

    def __len__(self):
        """Returns the number of active bits in current SDR."""
        return len(self.flat_index)

    def zero(self):
        """Sets all bits in the current sdr to zero."""
        self._dense      = None
        self._index      = None
        self._flat_index = np.empty(0, dtype=np.int)

    def assign_flat_concatenate(self, sdrs):
        """Flats and joins its inputs, assigns the result to its current value."""
        sdrs = tuple(sdrs)
        assert(all(isinstance(s, SDR) for s in sdrs))
        self._dense      = None
        self._index      = None
        self._flat_index = np.empty(0, dtype=np.int)
        offset = 0
        for sdr in sdrs:
            self._flat_index = np.concatenate([self._flat_index, sdr.flat_index + offset])
            offset += sdr.size
        self._handle_callbacks()

    def slice_into(self, sdrs):
        """
        This divides this SDR's current value into peices and gives a peice to
        each of the given SDRs.  All SDRs must be 1 dimension.
        """
        sdrs = tuple(sdrs)
        assert(all(isinstance(s, SDR) for s in sdrs))
        assert(len(self.dimensions) == 1)
        assert(all(len(s.dimensions) == 1 for s in sdrs))
        assert(sum(s.size for s in sdrs) == self.size)

        offset = 0
        for slice_sdr in sdrs:
            slice_sdr.dense = self.dense[offset: offset + slice_sdr.size]
            offset += slice_sdr.size

    def overlap(self, other_sdr):
        """
        Documentation ...
        Explain that this is a measure of semantic similarity between two SDRs.

        Argument other_sdr is assigned into an SDR with the same dimensions as
                 this SDR, see SDR.assign for more information.

        Returns a number in the range [0, 1]
        """
        other = SDR(self)
        other.assign(other_sdr)
        overlap_bits = np.count_nonzero(np.logical_and(self.dense, other.dense))
        average_bits = (len(self) + len(other)) / 2
        return overlap_bits / average_bits

    def false_positive_rate(self, active_sample_size, overlap_threshold):
        """
        Returns the theoretical false positive rate for a dendritic segment
        detecting the current value of this SDR.  This returns the probabilty
        random noise will activate the dendritic segment.

        Argument active_sample_size is the number of active bits which are
                 sampled onto the segment.
        Argument overlap_threshold is how many active bits are needed to
                 depolarize the segment.
        Argument self, this uses the current number of active bits, len(self) or
                 the mean activation frequency if it is available.

        Source: arXiv:1601.00720 [q-bio.NC], equation 4.
        """
        overlap_threshold = math.ceil(overlap_threshold)
        if hasattr(self, 'activation_frequency'):
            num_active_bits = int(round(np.mean(self.activation_frequency) * self.size))
        else:
            num_active_bits = len(self)
        assert(active_sample_size <= num_active_bits) # Can't sample more bits than are active.
        num_inactive_bits = self.size - num_active_bits
        # Overlap set size is number of possible values for this SDR which
        # this segment could falsely detect.
        overlap_size = 0
        for overlap in range(overlap_threshold, active_sample_size+1):
            overlap_size += (_choose(num_active_bits, overlap)
                           * _choose(num_inactive_bits, active_sample_size - overlap))
        overlap_size -= 1 # For the intended pattern.
        # Divide by the number of different things which this segment could
        # possibly detect.
        num_possible_detections = _choose(self.size, active_sample_size)
        return Fraction(overlap_size, num_possible_detections)

    def false_negative_rate(self, missing_activations, active_sample_size, overlap_threshold):
        """
        Returns the theoretical false negative rate for a dendritic segment
        detecting the current value of this SDR.  This returns the probability
        that a segment which would normally detect the value will fail when some
        activations are supressed.

        Argument missing_activations is the fraction of neuron activations which
                 are missing from this SDR.
        Argument active_sample_size is how many active inputs are used to detect
                 the value.
        Argument overlap_threshold is how many active inputs are needed to
                 depolarize the segment, resulting in a prediction.
        Argument self, this uses the current number of active bits, len(self) or
                 the mean activation frequency if it is available.

        Source: arXiv:1601.00720 [q-bio.NC], equation 6.
        """
        overlap_threshold = math.ceil(overlap_threshold)
        if hasattr(self, 'activation_frequency'):
            num_active_bits = int(round(np.mean(self.activation_frequency) * self.size))
        else:
            num_active_bits = len(self)
        assert(active_sample_size >= overlap_threshold) # Otherwise the segment would never activate.
        assert(active_sample_size <= num_active_bits) # Can't sample more bits than are active.
        assert(0 <= missing_activations <= 1) # missing_activations is the fraction of all activations.
        missing_activations = int(round(missing_activations * num_active_bits))
        # Count how many ways there are to corrupt this SDR such that it would
        # not be detected.
        false_negatives = 0
        # The overlap is between the corrupted bits and the segment's sample of
        # active bits.
        for overlap in range(min(missing_activations, active_sample_size)):
            if active_sample_size - overlap >= overlap_threshold:
                # There are too few corrupted bits in the active sample to
                # possibly cause a false negative.
                continue
            # The first part is the number of ways which corrupted bits could
            # fall in this segment.  The second part is the number of ways in
            # which corrupted bits could fall outside this segment.
            false_negatives += (_choose(active_sample_size, overlap)
                              * _choose(num_active_bits - active_sample_size, missing_activations - overlap))
        # Divide by the total number of ways to corrupt this SDR.
        num_corruptions = _choose(num_active_bits, missing_activations)
        return Fraction(false_negatives, num_corruptions)

    def entropy(self):
        """
        Calculates the entropy of this SDRs activations.

        Result is normalized to range [0, 1]
        A value of 1 indicates that all bits are equally and fully utilized.
        """
        if not hasattr(self, 'activation_frequency'):
            raise TypeError('Can not calculate entropy unless activation frequency is enabled for SDR.')
        p = self.activation_frequency
        e = _binary_entropy(p) / _binary_entropy(np.mean(p))
        return e

    def add_noise(self, percent):
        """Returns a copy of this SDR with the given percent of active bits moved."""
        assert(False) # Unimplemented.
        noisy     = SDR(self)
        # flip_off  = random.sample(on_bits, flip_bits)
        # flip_on   = random.sample(off_bits, flip_bits)
        # noisy.dense[list(zip(*flip_off))] = False
        # noisy.dense[list(zip(*flip_on))]  = True
        return noisy

    def statistics(self):
        """Returns a string describing this SDR."""
        stats = 'SDR%s\n'%str(self.dimensions)

        if hasattr(self, 'average_overlap'):
            stats += '\tAverage Overlap %g\n'%self.average_overlap

        if hasattr(self, 'activation_frequency'):
            af = self.activation_frequency
            stats += '\tEntropy: %d%%\n'%round(self.entropy()*100)
            stats += '\tActivation Frequency min/mean/std/max  %-.04g%% / %-.04g%% / %-.04g%% / %-.04g%%\n'%(
                np.min(af)*100,
                np.mean(af)*100,
                np.std(af)*100,
                np.max(af)*100,)
        else:
            stats += '\tCurrent Sparsity %.04g%%\n'%(100 * len(self) / self.size)

        return stats

SDR = SparseDistributedRepresentation


DEF DEBUG = True

ctypedef np.float32_t PERMANENCE_t  # For Cython's compile time type system.
PERMANENCE = np.float32             # For Python3's run time type system.

class SynapseManager:
    """
    This class models NMDA Receptor synapses.

    Internal to this class, all inputs and outputs are identified by their index
    into their flattened space.  This class indexes its synapses by both the
    presynaptic input index and the postsynaptic output index.  Each output
    keeps a list of potential inputs, each input keeps a list of potential
    outputs, and both sides contain the complete location of their other
    endpoint.  In this way every input and output can access all of its data in
    constant time.

    Attributes postsynaptic_sources, postsynaptic_permanences
        postsynaptic_sources[output-index] = 1D array of potential input indecies.
        postsynaptic_permanences[output-index] = 1D array of potential input permanences.
        These tables run parallel to each other.  These tables are the original
        data tables; the other tables are calculated from these two tables by
        the method rebuild_indexes().  These two tables specify the pool of
        potential and actual input connections to each output location, and are
        refered to as the 'potential_pool'.

    Attributes presynaptic_sinks
        presynaptic_sinks[input-index] = 1D array of connected output indecies.
        The sinks table is associates each input (presynapse) with its outputs
        (postsynapses), allowing for fast feed forward calculations.

    Attribute presynaptic_partitions
        Each entry in presynaptic_sinks is partitioned into connected and
        disconnected synapses.  This allows for fast feed forward calculations.
        In pseudocode:
            partition             = presynaptic_partitions[input-index]
            synapse_outputs       = presynaptic_sinks[input-index]
            connected_synapses    = synapse_outputs[0 : partition]
            disconnected_synapses = synapse_outputs[partition : ]

    Since postsynaptic_sources and presynaptic_sinks are arrays of arrays, two
    indecies are needed to go back and forth between them.  Attributes
    postsynaptic_source_side_index and presynaptic_sink_side_index are tables
    containing these second indexes.

    Attribute postsynaptic_source_side_index
        This table runs parallel to the postsynaptic_sources table and contains
        the second index into the presynaptic_sinks table.  Values in this table
        can be derived from the following pseudocode:
            output_index         = 1234
            potential_pool_index = 66
            input_index          = postsynaptic_sources[output_index][potential_pool_index]
            source_side_index    = presynaptic_sinks[input_index].index(output_index)  # See help(list.index)
            postsynaptic_source_side_index[output_index][potential_pool_index] = source_side_index

    Attribute presynaptic_sink_side_index
        This table runs parallel to presynaptic_sinks.  It contains the second
        index into the postsynaptic_sources table.  Values in this table can be
        derived from the following pseudocode:
            input_index     = 12345
            sink_number     = 99
            output_index    = presynaptic_sinks[input_index][sink_number]
            potential_index = postsynaptic_sources[output_index].index(input_index)
            presynaptic_sink_side_index[input_index][sink_number] = potential_index

    Data types:
        Network Size Assumptions:
            self.num_inputs  <= 2^32, input space is uint32 addressable.
            self.num_outputs <= 2^32, output space is uint32 addressable.

        postsynaptic_sources.dtype == np.uint32, index into input space.

        postsynaptic_source_side_index.dtype == np.uint32, maximum number of 
            outputs which an input could connect to, size of output space.

        presynaptic_sinks.dtype == np.uint32, index into output space.

        presynaptic_partitions.dtype == np.uint32, maximum number of outputs
            which an input could connect to at one time, size of output space.

        presynaptic_sink_side_index.dtype == np.uint32, maximum number of inputs
            which an output could have in its potential pool, size of input space.

        postsynaptic_permanences.dtype == float32.
        self.permanence_inc     is a python float, cast to float32 before using!
        self.permanence_dec     is a python float, cast to float32 before using!
        self.permanence_thresh  is a python float, cast to float32 before using!
        NOTE: Synapses are connected if and only if: permanence >= threshold.
        And the both variables MUST use the correct types!  There are NO checks
        on the validity of the internal data structures and corruption typically
        causes out of bounds array access errors.
    """
    def __init__(self, input_sdr, output_sdr, permanence_thresh,
        permanence_inc = 0,
        permanence_dec = 0,):
        """
        Argument input_sdr is the presynaptic input activations.
        Argument output_sdr is the postsynaptic segment sites.

        Argument permanence_thresh ...
        Optional Argument permanence_inc ...
        Optional Argument permanence_dec ...
        """
        assert(isinstance(input_sdr, SDR))
        assert(isinstance(output_sdr, SDR))
        self.inputs            = input_sdr
        self.outputs           = output_sdr
        self.permanence_inc    = PERMANENCE(permanence_inc)
        self.permanence_dec    = PERMANENCE(permanence_dec)
        self.permanence_thresh = PERMANENCE(permanence_thresh)

        assert(self.inputs.size  <= 2**32)   # self.postsynaptic_sources is a uint32 index into input space
        assert(self.outputs.size <= 2**32)   # self.presynaptic_sinks is a uint32 index into output space

        self.postsynaptic_sources       = np.empty(self.outputs.size, dtype=object)
        self.postsynaptic_permanences   = np.empty(self.outputs.size, dtype=object)
        for idx in range(self.outputs.size):
            # Note: np.empty(dtype=object) fills array with None.  It's faster
            # to initialize the tables have all valid (if empty) entries than to
            # check for None's at every turn.
            self.postsynaptic_sources[idx]     = np.empty((0,), dtype=np.uint32)
            self.postsynaptic_permanences[idx] = np.empty((0,), dtype=PERMANENCE)
        self.rebuild_indexes()  # Initializes sinks index et al.

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
    def compute(self, input_sdr=None):
        """
        Applies presynaptic activity to synapses, returns the postsynaptic
        excitment.

        Argument input_sdr ... is assigned to this classes internal inputs SDR.
                 If not given this uses the current value of its inputs SDR,
                 which this synapse manager was initialized with.

        Returns the excitement ... shape is output_sdr.dimensions
        """
        self.inputs.assign(input_sdr)

        # Cython assumes ndim=1 by default.
        cdef np.ndarray[np.int8_t]     inps        = self.inputs.dense.reshape(-1)
        cdef np.ndarray[np.uint32_t]   excitement  = np.zeros(self.outputs.size, dtype=np.uint32)
        cdef np.ndarray[dtype=object]  sinks_table = self.presynaptic_sinks
        cdef np.ndarray[np.uint32_t]   sinks_entry
        cdef np.ndarray[np.uint32_t]   sink_sizes  = self.presynaptic_partitions
        # cdef np.ndarray[np.int_t]      active_inputs = np.nonzero(self.input_activity)[0]
        cdef np.uint32_t inp_idx, out_idx
        cdef int iter1, iter2

        # TODO: Consider rolling the nonzero opperation directly into this loop
        #       Make sure it runs faster too, although currently is makes input dense and THEN takes the nonzero...
        # for iter1 in range(active_inputs.shape[0]):
        #     inp_idx = active_inputs[iter1]
        for inp_idx in range(inps.shape[0]):
            if inps[inp_idx] == 0:
                continue
            IF DEBUG:   # Safe cast (type checks and throws exceptions if needed)
                sinks_entry = <np.ndarray[np.uint32_t, ndim=1]?> sinks_table[inp_idx]
            ELSE:       # Unsafe Cast
                sinks_entry = <np.ndarray[np.uint32_t, ndim=1]> sinks_table[inp_idx]
            for iter2 in range(sink_sizes[inp_idx]):
                out_idx = sinks_entry[iter2]
                excitement[out_idx] += 1
        return excitement.reshape(self.outputs.dimensions)

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
    def learn_outputs(self, input_sdr=None, output_sdr=None,
        traces=None,
        permanence_inc = None,
        permanence_dec = None,):
        """
        Update the permanences of active outputs using the most recently given
        inputs.

        This method accepts one of the following arguments:
        Argument output_sdr ...
        Argument traces ... dense dtype=PERMANENCE

        Optional Arguments permanence_inc and permanence_dec take precedence over
                           any value passed to this classes initialize method.
        """
        self.inputs.assign(input_sdr)
        self.outputs.assign(output_sdr)
        cdef:
            # Data tables
            np.ndarray[dtype=object] sources          = self.postsynaptic_sources
            np.ndarray[dtype=object] sources2         = self.postsynaptic_source_side_index
            np.ndarray[dtype=object] permanences      = self.postsynaptic_permanences
            np.ndarray[dtype=object] sinks            = self.presynaptic_sinks
            np.ndarray[dtype=object] sinks2           = self.presynaptic_sink_side_index
            np.ndarray[np.uint32_t]  sink_partitions  = self.presynaptic_partitions
            np.ndarray[np.int_t]     output_activity  = self.outputs.flat_index
            np.ndarray[PERMANENCE_t] traces_          = traces
            np.ndarray[np.int8_t]    input_activity   = self.inputs.dense.reshape(-1)

            # Inner array pointers
            np.ndarray[np.uint32_t]  sources_inner
            np.ndarray[np.uint32_t]  sources2_inner
            np.ndarray[np.uint32_t]  sources2_inner_swap
            np.ndarray[PERMANENCE_t] perms_inner
            np.ndarray[np.uint32_t]  sinks_inner
            np.ndarray[np.uint32_t]  sinks2_inner

            # Indexes and locals
            int out_loop_size
            np.uint32_t out_iter, out_idx1, out_idx2
            np.uint32_t inp_idx1, inp_idx2, inp_idx2_swap
            PERMANENCE_t perm_value
            PERMANENCE_t inc, dec, thresh = self.permanence_thresh
            PERMANENCE_t trace
            PERMANENCE_t weighted_inc
            PERMANENCE_t weighted_dec
            bint syn_prior, syn_post
            np.uint32_t temp_sink, temp_sink2

        # Arguments override initialized or default values.
        inc = permanence_inc if permanence_inc is not None else self.permanence_inc
        dec = permanence_dec if permanence_dec is not None else self.permanence_dec
        weighted_inc = inc
        weighted_dec = dec
        if inc == 0. and dec == 0.:
            return

        # This loop works for two loop variants, indexes and traces.
        if traces_ is None:
            out_loop_size = output_activity.shape[0]
        else:
            out_loop_size = traces_.shape[0]
            assert(traces_.shape[0] == self.outputs.dimensions)

        for out_iter in range(out_loop_size):
            if traces_ is None:
                out_idx1    = output_activity[out_iter]
            else:
                trace       = traces_[out_iter]
                if trace == 0.:
                    continue
                weighted_inc = inc * trace
                weighted_dec = dec * trace
                out_idx1    = out_iter

            # Unsafe type cast.
            sources_inner   = <np.ndarray[np.uint32_t, ndim=1]>  sources[out_idx1]
            sources2_inner  = <np.ndarray[np.uint32_t, ndim=1]>  sources2[out_idx1]
            perms_inner     = <np.ndarray[PERMANENCE_t, ndim=1]> permanences[out_idx1]

            for out_idx2 in range(sources_inner.shape[0]):
                inp_idx1     = sources_inner[out_idx2]
                perm_value   = perms_inner[out_idx2]

                syn_prior    = perm_value >= thresh
                if input_activity[inp_idx1] != 0:
                    perm_value += weighted_inc
                    if perm_value > 1.:
                        perm_value = 1.
                else:
                    perm_value -= weighted_dec
                    if perm_value < 0.:
                        perm_value = 0.
                syn_post      = perm_value >= thresh
                perms_inner[out_idx2] = perm_value

                if syn_prior != syn_post:
                    inp_idx2     = sources2_inner[out_idx2]
                    sinks_inner  = <np.ndarray[np.uint32_t, ndim=1]> sinks[inp_idx1]
                    sinks2_inner = <np.ndarray[np.uint32_t, ndim=1]> sinks2[inp_idx1]
                    # Repartition the sinks table.  First determine
                    # inp_idx2_swap, which is the synapse to swap with.
                    if syn_post:
                        # Connect this synapse.  Swap to the bottom of the
                        # disconnected partition and then increment the
                        # partition across the now connected synapse.
                        inp_idx2_swap = sink_partitions[inp_idx1]
                        sink_partitions[inp_idx1] = inp_idx2_swap + 1
                    else:
                        # Disconnect this synapse.  Swap to end of the connected
                        # partition and then decrement the partition across the
                        # now disconnected synapse.
                        inp_idx2_swap = sink_partitions[inp_idx1] - 1
                        sink_partitions[inp_idx1] = inp_idx2_swap

                    # Do the swap.  Notify the postsynaptic side of the synapse
                    # which got preempted.
                    out_idx_1_swap = sinks_inner[inp_idx2_swap]
                    out_idx_2_swap = sinks2_inner[inp_idx2_swap]
                    sources2_inner_swap = <np.ndarray[np.uint32_t, ndim=1]> sources2[out_idx_1_swap]
                    sources2_inner_swap[out_idx_2_swap] = inp_idx2
                    # Notify the postsynaptic side of the synapse which was
                    # moved across the partition.
                    sources2_inner[out_idx2] = inp_idx2_swap

                    # Swap the synapses in the sinks tables.
                    temp_sink                   = sinks_inner[inp_idx2_swap]
                    sinks_inner[inp_idx2_swap]  = sinks_inner[inp_idx2]
                    sinks_inner[inp_idx2]       = temp_sink
                    # Second swap for the inner index.
                    temp_sink2                  = sinks2_inner[inp_idx2_swap]
                    sinks2_inner[inp_idx2_swap] = sinks2_inner[inp_idx2]
                    sinks2_inner[inp_idx2]      = temp_sink2
    learn = learn_outputs

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
    def learn_inputs(self, input_sdr=None, output_sdr=None, permanence_inc=None, permanence_dec=None):
        """
        Update permanences and then synapses.

        Instead of decrementing permanences when then output fires without
        the input, decrement when the input fires and the output doesn't.
        This is the difference bewtween P implies Q and Q implies P, or in
        this context:
            neuron activation -> Proximal/Basal input
              and
            Apical input -> neuron activation
        """
        """
        I think the idea was that this different learning rule would 'protect'
        synapses which are not often used.  The thing is there is already a
        protection mechanism...  Synapses only learn if their segment meets the
        learning threshold which protects segments when they aren't in use.
        BG presynaptic sources only need to worry about unlearning things which
        they're currently trying to do.  Don't use this method?
        """
        self.inputs.assign(input_sdr)
        self.outputs.assign(output_sdr)

        cdef PERMANENCE_t perm_value, inc, dec, thresh = self.permanence_thresh
        cdef bint syn_prior, syn_post
        cdef np.uint32_t inp_idx1, inp_idx2, out_idx1, out_idx2
        cdef np.int8_t inp, out
        cdef bint loop_skip
        cdef np.uint32_t original_partition


        inc = permanence_inc if permanence_inc is not None else self.permanence_inc
        dec = permanence_dec if permanence_dec is not None else self.permanence_dec

        cdef:
            # Data Tables
            np.ndarray[dtype=object] sources          = self.postsynaptic_sources
            np.ndarray[dtype=object] sources2         = self.postsynaptic_source_side_index
            np.ndarray[dtype=object] permanences      = self.postsynaptic_permanences
            np.ndarray[dtype=object] sinks            = self.presynaptic_sinks
            np.ndarray[dtype=object] sinks2           = self.presynaptic_sink_side_index
            np.ndarray[np.uint32_t]  sink_partitions  = self.presynaptic_partitions
            np.ndarray[np.int8_t]    input_activity   = self.inputs.dense.reshape(-1)
            np.ndarray[np.int8_t]    dense_output     = self.outputs.dense.reshape(-1)
            # Swap Variables
            np.uint32_t temp_sink, temp_sink2, inp_idx2_swap, out_idx_1_swap, out_idx_2_swap
            np.ndarray[np.uint32_t]  sinks_inner
            np.ndarray[np.uint32_t]  sinks2_inner

        # Iterate through the active inputs.
        for inp_idx1 in range(input_activity.shape[0]):
            inp = input_activity[inp_idx1]
            if inp == 0:
                continue

            # Make sure not to process newly disconnected synapses twice. They
            # are moved into the range between the start of the disconnected
            # partition and where the start of said partition was when this
            # method began.  Skip this section of the loop.
            original_partition = sink_partitions[inp_idx1]
            # Do the loop skip unless it would skip off the end of the array.
            loop_skip          = original_partition < sinks[inp_idx1].shape[0]
            for inp_idx2 in range(sinks[inp_idx1].shape[0]):
                if inp_idx2 == sink_partitions[inp_idx1]:
                    if loop_skip:
                        inp_idx2  = original_partition
                        loop_skip = False   # Don't do this step twice.

                # Modify the synapses permanence.
                out_idx1    = sinks [inp_idx1][inp_idx2]
                out_idx2    = sinks2[inp_idx1][inp_idx2]
                out         = dense_output[out_idx1]
                perm_value  = permanences[out_idx1][out_idx2]
                syn_prior   = perm_value >= thresh
                if out != 0:
                    perm_value += inc
                    if perm_value > 1.:
                        perm_value = 1.
                else:
                    perm_value -= dec
                    if perm_value < 0.:
                        perm_value = 0.
                syn_post = perm_value >= thresh
                permanences[out_idx1][out_idx2] = perm_value

                if syn_prior != syn_post:
                    # Repartition the sinks table.  First determine
                    # inp_idx2_swap, which is the synapse to swap with.
                    if syn_post:
                        # Connect this synapse.  Swap to the bottom of the
                        # disconnected partition and then increment the
                        # partition across the now connected synapse.
                        inp_idx2_swap = sink_partitions[inp_idx1]
                        sink_partitions[inp_idx1] = inp_idx2_swap + 1
                    else:
                        # Disconnect this synapse.  Swap to end of the connected
                        # partition and then decrement the partition across the
                        # now disconnected synapse.
                        inp_idx2_swap = sink_partitions[inp_idx1] - 1
                        sink_partitions[inp_idx1] = inp_idx2_swap

                    # Do the swap.  Notify the postsynaptic side of the synapse
                    # which got preempted.
                    out_idx_1_swap = sinks [inp_idx1][inp_idx2_swap]
                    out_idx_2_swap = sinks2[inp_idx1][inp_idx2_swap]
                    sources2[out_idx_1_swap][out_idx_2_swap] = inp_idx2
                    # Notify the postsynaptic side of the synapse which was
                    # moved across the partition.
                    sources2[out_idx1][out_idx2] = inp_idx2_swap

                    # Swap the synapses in the sinks tables.
                    sinks_inner                 = sinks[inp_idx1]
                    temp_sink                   = sinks_inner[inp_idx2_swap]
                    sinks_inner[inp_idx2_swap]  = sinks_inner[inp_idx2]
                    sinks_inner[inp_idx2]       = temp_sink
                    # Second swap for the inner index.
                    sinks2_inner                = sinks2[inp_idx1]
                    temp_sink2                  = sinks2_inner[inp_idx2_swap]
                    sinks2_inner[inp_idx2_swap] = sinks2_inner[inp_idx2]
                    sinks2_inner[inp_idx2]      = temp_sink2

                    if not syn_post:
                        # Decrement the loop counter so that the preempted
                        # synapse gets processed instead of skipped over.
                        inp_idx2 -= 1

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
    def add_synapses(self, input_sdr=None, output_sdr=None, synapses_per_output=None, maximum_synapses=None,
        init_value=None):
        """
        Argument inputs is index array ...
        Argument outputs is index array ...
        Argument synapses_per_output is number ...
        Argument maximum_synapses is per output ...
        """
        self.inputs.assign(input_sdr)
        self.outputs.assign(output_sdr)
        cdef np.ndarray[np.uint32_t] inputs_ = np.array(self.inputs.flat_index, dtype=np.uint32)
        cdef np.ndarray[np.long_t] outputs_  = self.outputs.flat_index
        if not len(inputs_):
            return

        cdef int synapses_per_output_  = int(round(synapses_per_output))
        cdef int maximum_synapses_     = int(round(maximum_synapses))
        cdef PERMANENCE_t perm, thresh = self.permanence_thresh

        cdef:
            # Data Tables.
            np.ndarray[dtype=object] sources          = self.postsynaptic_sources
            np.ndarray[dtype=object] sources2         = self.postsynaptic_source_side_index
            np.ndarray[dtype=object] permanences      = self.postsynaptic_permanences
            np.ndarray[dtype=object] sinks            = self.presynaptic_sinks
            np.ndarray[dtype=object] sinks2           = self.presynaptic_sink_side_index
            np.ndarray[np.uint32_t]  sink_partitions  = self.presynaptic_partitions
            # Swap Variables.
            np.uint32_t temp_sink, temp_sink2, inp_idx2_swap, out_idx_1_swap, out_idx_2_swap
            np.ndarray[np.uint32_t]  sources_inner
            np.ndarray[np.uint32_t]  sinks_inner
            np.ndarray[np.uint32_t]  sinks2_inner
            # Local indexes and counters.
            int num_synapses
            int out_iter, out_idx, out_idx2
            np.uint32_t inp, inp_idx2
            int old_size, offset
            np.ndarray[np.uint32_t]  candidate_sources
            # np.ndarray[np.bool_t]    unique_sources
            np.ndarray[np.uint32_t]  new_sources
            np.ndarray[PERMANENCE_t] new_permanances

        for out_iter in range(outputs_.shape[0]):
            out_idx           = outputs_[out_iter]
            sources_inner     = sources[out_idx]
            old_size          = sources_inner.shape[0]

            # Randomly select inputs which are not already connected to.
            num_synapses      = min(synapses_per_output_, maximum_synapses_ - old_size, inputs_.shape[0])
            candidate_sources = np.random.choice(inputs_, (num_synapses,), replace=False)
            unique_sources    = np.isin(candidate_sources, sources_inner, assume_unique=True, invert=True)
            new_sources       = candidate_sources[unique_sources]
            # Append to the sources & permanence tables.
            if init_value is None:
                new_permanances  = np.array(np.random.uniform(size=new_sources.shape[0]), dtype=PERMANENCE)
            else:
                new_permanances  = np.empty(new_sources.shape[0], dtype=PERMANENCE)
                new_permanances.fill(init_value)
            sources[out_idx]     = np.append(sources_inner,        new_sources)
            permanences[out_idx] = np.append(permanences[out_idx], new_permanances)
            sources2[out_idx]    = np.append(sources2[out_idx],    np.empty(new_sources.shape[0], dtype=np.uint32))
            # Insert new synapses into the sinks tables.
            for offset in range(new_sources.shape[0]):
                inp          = new_sources[offset]
                perm         = new_permanances[offset]
                out_idx2     = old_size + offset
                sinks_inner  = sinks[inp]
                sinks2_inner = sinks2[inp]
                inp_idx2     = sinks_inner.shape[0] # Append and swap if needed.
                sinks[inp]   = sinks_inner  = np.append(sinks_inner, np.uint32(out_idx))
                sinks2[inp]  = sinks2_inner = np.append(sinks2_inner, np.uint32(out_idx2))
                sources2[out_idx][out_idx2] = inp_idx2
                if perm >= thresh:
                    # Connect this synapse.  Swap to the bottom of the
                    # disconnected partition and then increment the partition
                    # across the now connected synapse.
                    inp_idx2_swap = sink_partitions[inp]
                    sink_partitions[inp] = inp_idx2_swap + 1

                    # Do the swap.  Notify the postsynaptic side of the synapse
                    # which got preempted.
                    out_idx_1_swap = sinks_inner[inp_idx2_swap]
                    out_idx_2_swap = sinks2_inner[inp_idx2_swap]
                    sources2[out_idx_1_swap][out_idx_2_swap] = inp_idx2
                    # Notify the postsynaptic side of the synapse which was
                    # moved across the partition.
                    sources2[out_idx][out_idx2] = inp_idx2_swap

                    # Swap the synapses in the sinks tables.
                    temp_sink                   = sinks_inner[inp_idx2_swap]
                    sinks_inner[inp_idx2_swap]  = sinks_inner[inp_idx2]
                    sinks_inner[inp_idx2]       = temp_sink
                    # Second swap for the inner index.
                    temp_sink2                  = sinks2_inner[inp_idx2_swap]
                    sinks2_inner[inp_idx2_swap] = sinks2_inner[inp_idx2]
                    sinks2_inner[inp_idx2]      = temp_sink2

    def normally_distributed_connections(self, potential_pool, radii):
        """
        Makes synapses from inputs to outputs within their local neighborhood.
        The outputs exist on a uniform grid which is stretched over the input
        space.  An outputs local neighborhood is defined by a gaussian window
        centered over the output with standard deviations given by argument
        radii.

        Argument radii is tuple of standard deivations. Radii units are the
                 input space units.  Radii defines the topology of the
                 connections.  If radii are shorter than the number of input or
                 output space dimensions then the trailing input dimensions are
                 not considered topological dimensions.  These 'extra'
                 dimensions are treated with uniform probability; only distances
                 in the topological dimensions effect the probability of forming
                 a potential synapse.

        Argument potential_pool is the number of potential inputs to connect
                 each output to.

        Attributes set by this method:
            self.potential_pool_density_1,
            self.potential_pool_density_2,
            self.potential_pool_density_3,
                These measure the average fraction of inputs which are
                potentially connected to each outputs, looking within the first
                three standard deviations of the outputs receptive field.  The
                areas are non-overlapping.  These are incorperated into the
                statistics method if they are available.

        Returns inhibition_radii which is the is the radii after converting it
                into output space units.

        Note: this method does NOT check for duplicate synapses and should only
        be called on an empty synapse manager with no existing synapses.
        """
        radii = np.array(radii)
        assert(len(radii.shape) == 1)
        potential_pool = int(round(potential_pool))

        # Split the input space into topological and extra dimensions.
        topo_dimensions  = self.inputs.dimensions[: len(radii)]
        extra_dimensions = self.inputs.dimensions[len(radii) :]
        topo_output_dims = self.outputs.dimensions[: len(radii)]

        # Density Statistics
        potential_pool_density_1 = 0
        potential_pool_density_2 = 0
        potential_pool_density_3 = 0
        extra_area   = np.product(extra_dimensions)
        num_inputs_1 = extra_area * math.pi * np.product(radii)
        num_inputs_2 = extra_area * math.pi * np.product(2 * radii)
        num_inputs_3 = extra_area * math.pi * np.product(3 * radii)
        num_inputs_2 -= num_inputs_1
        num_inputs_3 -= num_inputs_1 + num_inputs_2

        # Find where the columns are in the input.
        output_ranges     = [slice(0, size) for size in self.outputs.dimensions]
        output_index      = np.mgrid[output_ranges]
        # output_locations[input-dimension][output-index] = Location in input
        # space.  This does not hold extra dimensions, only topological ones.
        output_locations  = [dim.flatten() for dim in output_index[: len(radii)]]
        padding           = radii   # No wrapping.
        expand_to         = np.subtract(topo_dimensions, np.multiply(2, padding))
        column_spacing    = np.divide(expand_to, topo_output_dims).reshape(len(topo_dimensions), 1)
        output_locations *= column_spacing
        output_locations += np.array(padding).reshape(len(topo_dimensions), 1)
        inhibition_radii  = radii / np.squeeze(column_spacing)

        for column_index in range(self.outputs.size):
            center = output_locations[:, column_index]
            # Make potential-pool many unique input locations.  This is an
            # iterative process: sample the normal distribution, reject
            # duplicates, repeat until done.  Working pool holds the
            # intermediate input-coordinates until it's filled and ready to be
            # spliced into self.postsynaptic_sources[column-index, :]
            working_pool  = np.empty((0, len(self.inputs.dimensions)), dtype=np.int)
            empty_sources = potential_pool  # How many samples to take.
            for attempt in range(10):
                # Sample points from the input space and cast to valid indecies.
                # Take more samples than are needed B/C some will not be viable.
                topo_pool     = np.random.normal(center, radii, 
                                    size=(max(256, 2*empty_sources), len(radii)))
                topo_pool     = np.rint(topo_pool)   # Round towards center
                # Discard samples which fall outside of the input space.
                out_of_bounds = np.logical_or(topo_pool < 0, topo_pool >= topo_dimensions)
                out_of_bounds = np.any(out_of_bounds, axis=1)
                topo_pool     = topo_pool[np.logical_not(out_of_bounds)]
                extra_pool    = np.random.uniform(0, extra_dimensions,
                                size=(topo_pool.shape[0], len(extra_dimensions)))
                extra_pool    = np.floor(extra_pool) # Round down to stay in half open range [0, dim)
                # Combine topo & extra dimensions into input space coordinates.
                pool          = np.concatenate([topo_pool, extra_pool], axis=1)
                pool          = np.array(pool, dtype=np.int)
                # Add the points to the working pool.
                working_pool  = np.concatenate([working_pool, pool], axis=0)
                # Reject duplicates.
                working_pool  = np.unique(working_pool, axis=0)
                empty_sources = potential_pool - working_pool.shape[0]
                if empty_sources <= 0:
                    break
            else:
                if empty_sources > .05 * potential_pool:
                    raise ValueError("Not enough sources to fill potential pool.")
                else:
                    print("Warning: Could not find enough unique inputs, allowing %d fewer inputs..."%empty_sources)
            working_pool = working_pool[:potential_pool, :] # Discard extra samples

            # Measure some statistics about input density.
            displacements = working_pool[:, :len(topo_dimensions)] - center
            # Measure in terms of standard deviations of their distribution.
            deviations = displacements / radii
            distances  = np.sum(deviations**2, axis=1)**.5
            pp_size_1  = np.count_nonzero(distances <= 1)
            pp_size_2  = np.count_nonzero(np.logical_and(distances > 1, distances <= 2))
            pp_size_3  = np.count_nonzero(np.logical_and(distances > 2, distances <= 3))
            potential_pool_density_1 += pp_size_1 / num_inputs_1
            potential_pool_density_2 += pp_size_2 / num_inputs_2
            potential_pool_density_3 += pp_size_3 / num_inputs_3

            # Flatten and write to output array.
            working_pool = np.ravel_multi_index(working_pool.T, self.inputs.dimensions)
            working_pool = np.array(working_pool, dtype=np.uint32)
            initial_permanences = np.random.uniform(size=(len(working_pool),))
            initial_permanences = np.array(initial_permanences, dtype=PERMANENCE)
            self.postsynaptic_sources[column_index] = np.append(self.postsynaptic_sources[column_index], working_pool)
            self.postsynaptic_permanences[column_index] = np.append(self.postsynaptic_permanences[column_index], initial_permanences)

        self.rebuild_indexes()

        self.potential_pool_density_1 = potential_pool_density_1 / self.outputs.size
        self.potential_pool_density_2 = potential_pool_density_2 / self.outputs.size
        self.potential_pool_density_3 = potential_pool_density_3 / self.outputs.size
        return inhibition_radii

    def uniformly_distributed_connections(self, potential_pool):
        """
        Connect every output to potential_pool inputs.
        Directly sets the sources and permanence arrays, no returned value.
        Will raise ValueError if potential_pool is invalid.

        Note: this method does NOT check for duplicate synapses and should only
        be called on an empty synapse manager with no existing synapses.
        """
        potential_pool = int(round(potential_pool))
        for out_idx in range(self.outputs.size):
            syn_src = np.random.choice(self.inputs.size, potential_pool, replace=False)
            syn_src = np.array(syn_src, dtype=np.uint32)
            syn_prm = np.random.uniform(size=syn_src.shape)
            syn_prm = np.array(syn_prm, dtype=PERMANENCE)
            self.postsynaptic_sources[out_idx]     = np.append(self.postsynaptic_sources[out_idx], syn_src)
            self.postsynaptic_permanences[out_idx] = np.append(self.postsynaptic_permanences[out_idx], syn_prm)
        self.rebuild_indexes()

    def rebuild_indexes(self):
        """
        This method uses the postsynaptic_sources and postsynaptic_permanences
        tables to rebuild all of the other needed tables.
        """
        self.presynaptic_sinks              = sinks   = np.empty(self.inputs.size,  dtype=object)
        self.presynaptic_sink_side_index    = sinks2  = np.empty(self.inputs.size,  dtype=object)
        presynaptic_permanences             = perms   = np.empty(self.inputs.size,  dtype=object)
        self.presynaptic_partitions         = parts   = np.zeros(self.inputs.size,  dtype=np.uint32)
        self.postsynaptic_source_side_index = srcs2   = np.empty(self.outputs.size, dtype=object)

        cdef int inp_idx, out_idx, synapse_num

        # Initialize the index tables.
        for inp_idx in range(self.inputs.size):
            sinks[inp_idx]  = []    # Init with python lists for fast append.
            sinks2[inp_idx] = []
            perms[inp_idx]  = []
        for out_idx in range(self.outputs.size):
            assert(self.postsynaptic_sources[out_idx].dtype == np.uint32)
            num_sources     = len(self.postsynaptic_sources[out_idx])
            srcs2[out_idx]  = np.empty(num_sources, dtype=np.uint32)

        # Iterate through every synapse, build the presynaptic_sinks table.
        for out_idx in range(self.outputs.size):
            sources_inner = self.postsynaptic_sources[out_idx]
            perms_inner   = self.postsynaptic_permanences[out_idx]
            for synapse_num in range(sources_inner.shape[0]):
                inp_idx  = sources_inner[synapse_num]
                perm_val = perms_inner[synapse_num]
                sinks[inp_idx].append(out_idx)
                sinks2[inp_idx].append(synapse_num)
                perms[inp_idx].append(perm_val)

        # Partition the presynaptic_sinks table, also cast everything to numpy
        # arrays.
        for inp_idx in range(self.inputs.size):
            permanences     = np.array(presynaptic_permanences[inp_idx], dtype=PERMANENCE)
            sort_order      = np.argsort(-permanences)
            thresh          = PERMANENCE(self.permanence_thresh)
            parts[inp_idx]  = np.count_nonzero(permanences >= thresh)
            unsorted_sinks  = np.array(sinks[inp_idx], dtype=np.uint32)
            sinks[inp_idx]  = unsorted_sinks[sort_order]
            unsorted_sinks2 = np.array(sinks2[inp_idx], dtype=np.uint32)
            sinks2[inp_idx] = unsorted_sinks2[sort_order]
            # Notify the output side about where each synapse ended up.
            for synapse_num in range(sinks[inp_idx].shape[0]):
                out_idx_1 = sinks[inp_idx][synapse_num]
                out_idx_2 = sinks2[inp_idx][synapse_num]
                srcs2[out_idx_1][out_idx_2] = synapse_num

    def copy(self):
        """
        Makes a shallow copy of the synapse manager and its input/output SDRs,
        which effectively freezes and disconnects this from the rest of the
        system while sharing the same underlying data tables.  When the copy
        learns, the original also learns.
        """
        cpy = copy.copy(self)
        cpy.inputs  = SDR(self.inputs)
        cpy.outputs = SDR(self.outputs)
        return cpy

    def statistics(self):
        # TODO: DOCSTRING.  Add short explainations of what presynaptic,
        # postsynaptic, potential and connected mean.
        stats = 'Synapse Manager Statistics:\n'
        stats += 'Input ' + self.inputs.statistics()
        stats += 'Output ' + self.outputs.statistics()
        if hasattr(self, 'potential_pool_density_1'):
            stats += 'Potential Pool Density 1/2/3: {:5g} / {:5g} / {:5g}\n'.format(
                self.potential_pool_density_1,
                self.potential_pool_density_2,
                self.potential_pool_density_3,)

        # Build a small table of min/mean/std/max for pre & post, potential &
        # connected synapse counts (16 values total).
        presyn_potential  = [pp.shape[0] for pp in self.presynaptic_sinks]
        presyn_connected  = self.presynaptic_partitions
        postsyn_potential = [pp.shape[0] for pp in self.postsynaptic_sources]
        threshold         = PERMANENCE(self.permanence_thresh)
        postsyn_connected = [np.count_nonzero(pp_perm >= threshold)
                                for pp_perm in self.postsynaptic_permanences]
        entries = [
            (' '*8 + "Presyn sinks    Potential", presyn_potential),
            (' '*8 + "Postsyn sources Potential", postsyn_potential),
            (' '*8 + "Presyn sinks    Connected", presyn_connected),
            (' '*8 + "Postsyn sources Connected", postsyn_connected),
        ]
        header  = ' '*8 + 'Synapse Counts'
        header += ' '*(len(entries[0][0]) - len(header))
        header += ''.join([' | %5s'%c for c in ['min', 'mean','std', 'max']]) + '\n'
        stats  += header
        for name, data in entries:
            columns = ( name,
                        int(round(np.min(data))),
                        int(round(np.mean(data))),
                        int(round(np.std(data))),
                        int(round(np.max(data))),)
            stats += '{} | {: >5d} | {: >5d} | {: >5d} | {: >5d}\n'.format(*columns)

        return stats


class WeightedSynapseManager(SynapseManager):
    """
    This class models weighted synapses.

    Attributes postsynaptic_permanences, presynaptic_permanences contain the
               synapse strength.  The compute method adds the synapse strength 
               to the excitement accumulator instead of incrementing it.  The
               learn method still modifies the permanence using hebbian learning.
    """
    def rebuild_indexes(self):
        super().rebuild_indexes()
        self.presynaptic_permanences = np.empty(self.inputs.size, dtype=object)
        for idx in range(self.inputs.size):
            num_sinks = self.presynaptic_sinks[idx].shape[0]
            self.presynaptic_permanences[idx] = np.empty(num_sinks, dtype=np.float32)
            self.presynaptic_permanences[idx].fill(float('nan'))
        for out_idx1 in range(self.outputs.size):
            num_sources = self.postsynaptic_sources[out_idx1].shape[0]
            for out_idx2 in range(num_sources):
                inp_idx1 = self.postsynaptic_sources[out_idx1][out_idx2]
                inp_idx2 = self.postsynaptic_source_side_index[out_idx1][out_idx2]
                self.presynaptic_permanences[inp_idx1][inp_idx2]

    @cython.boundscheck(DEBUG) # Turns off bounds-checking for entire function.
    @cython.wraparound(DEBUG)  # Turns off negative index wrapping for entire function.
    def compute(self, input_sdr=None):
        """
        Applies presynaptic activity to synapses, returns the postsynaptic
        excitment.

        Argument input_sdr ... is assigned to this classes internal inputs SDR.
                 If not given this uses the current value of its inputs SDR,
                 which this synapse manager was initialized with.

        Returns the excitement ... shape is output_sdr.dimensions
        """
        self.inputs.assign(input_sdr)

        cdef:
            np.ndarray[np.int_t]      active_inputs = self.inputs.flat_index
            np.ndarray[dtype=object]  sinks_table   = self.presynaptic_sinks
            np.ndarray[np.uint32_t]   sinks_entry
            np.ndarray[dtype=object]  perms_table   = self.presynaptic_permanences
            np.ndarray[PERMANENCE_t]  perms_entry
            np.ndarray[np.float32_t]  excitement    = np.zeros(self.outputs.size, dtype=np.float32)
            np.uint32_t inp_idx1, inp_idx2, out_idx
            int iter1, iter2

        for iter1 in range(active_inputs.shape[0]):
            inp_idx1 = active_inputs[iter1]
            IF DEBUG:   # Safe cast (type checks and throws exceptions if needed)
                sinks_entry = <np.ndarray[np.uint32_t]?>  sinks_table[inp_idx1]
                perms_entry = <np.ndarray[PERMANENCE_t]?> perms_table[inp_idx1]
            ELSE:       # Unsafe Cast
                sinks_entry = <np.ndarray[np.uint32_t]>  sinks_table[inp_idx1]
                perms_entry = <np.ndarray[PERMANENCE_t]> perms_table[inp_idx1]
            for inp_idx2 in range(sinks_entry.shape[0]):
                out_idx = sinks_entry[inp_idx2]
                excitement[out_idx] += perms_entry[inp_idx2]
        return excitement.reshape(self.outputs.dimensions)

    def learn_outputs(self, input_sdr=None, output_sdr=None,
        permanence_inc = None,
        permanence_dec = None,):
        """
        Wrapper around SynapseManager.learn_outputs() which updates the
        presynaptic_permanences table.
        """
        super().learn_outputs(input_sdr=input_sdr, output_sdr=output_sdr,
            permanence_inc = permanence_inc,
            permanence_dec = permanence_dec)

        cdef:
            np.ndarray[np.int_t] output_activity = self.outputs.flat_index

            # Data tables
            np.ndarray[dtype=object] sources          = self.postsynaptic_sources
            np.ndarray[dtype=object] sources2         = self.postsynaptic_source_side_index
            np.ndarray[dtype=object] permanences      = self.postsynaptic_permanences
            np.ndarray[dtype=object] presyn_perms     = self.presynaptic_permanences

            # Inner array pointers
            np.ndarray[np.uint32_t]  sources_inner
            np.ndarray[np.uint32_t]  sources2_inner
            np.ndarray[PERMANENCE_t] perms_inner
            np.ndarray[PERMANENCE_t] presyn_perms_inner

            # Locals
            int out_iter, out_idx1, out_idx2
            int inp_idx1, inp_idx2

        for out_iter in range(len(self.outputs)):
            out_idx1 = output_activity[out_iter]

            # Unsafe type cast.
            sources_inner   = <np.ndarray[np.uint32_t, ndim=1]>  sources[out_idx1]
            sources2_inner  = <np.ndarray[np.uint32_t, ndim=1]>  sources2[out_idx1]
            perms_inner     = <np.ndarray[PERMANENCE_t, ndim=1]> permanences[out_idx1]

            for out_idx2 in range(sources_inner.shape[0]):
                inp_idx1 = sources_inner[out_idx2]
                inp_idx2 = sources2_inner[out_idx2]
                presyn_perms_inner = <np.ndarray[PERMANENCE_t]> presyn_perms[inp_idx1]
                presyn_perms_inner[inp_idx2] = perms_inner[out_idx2]

    def add_synapses(self, *args, **kw_args):
        super().add_synapses(*args, **kw_args)

        # Update the presynaptic-permanences table.
        cdef:
            # Data Tables.
            np.ndarray[dtype=object]  sinks         = self.presynaptic_sinks
            np.ndarray[np.uint32_t]   sinks_inner
            np.ndarray[dtype=object]  sinks2        = self.presynaptic_sink_side_index
            np.ndarray[np.uint32_t]   sinks2_inner
            np.ndarray[dtype=object]  presyn_perms  = self.presynaptic_permanences
            np.ndarray[PERMANENCE_t]  presyn_perms_inner
            np.ndarray[dtype=object]  postsyn_perms = self.postsynaptic_permanences
            np.ndarray[PERMANENCE_t]  postsyn_perms_inner

            # Locals.
            int inp_idx1, inp_idx2
            int num_sinks, num_perms
            PERMANENCE_t perm

        # Find all inputs which gained more sinks, but have not updated permanences.
        for inp_idx1 in range(self.inputs.size):
            sinks_inner        = sinks[inp_idx1]
            presyn_perms_inner = presyn_perms[inp_idx1]
            num_sinks   = sinks_inner.shape[0]
            num_perms   = presyn_perms_inner.shape[0]
            if num_sinks == num_perms:
                continue
            # Add space for the additional permanences.
            presyn_perms_inner = np.concatenate([
                presyn_perms_inner,
                np.zeros(num_sinks - num_perms, dtype=PERMANENCE)])
            presyn_perms[inp_idx1] = presyn_perms_inner
            # Find and store the new permanence values.
            sinks2_inner = sinks2[inp_idx1]
            for inp_idx2 in range(num_perms, num_sinks):
                out_idx1 = sinks_inner[inp_idx2]
                out_idx2 = sinks2_inner[inp_idx2]
                postsyn_perms_inner = postsyn_perms[out_idx1]
                perm = postsyn_perms_inner[out_idx2]
                presyn_perms_inner[inp_idx2] = perm


class Dendrite:
    """
    Makes predictions.
    """
    def __init__(self, input_sdr, active_sdr,
        segments_per_cell,
        synapses_per_segment,
        predictive_threshold,
        learning_threshold,
        permanence_thresh,
        permanence_inc,
        permanence_dec,
        mispredict_dec,
        add_synapses,
        initial_segment_size,):
        """!"""
        # TODO: copy the docstrings from synapse manager since this is a pass through
        assert(isinstance(input_sdr, SDR))
        assert(isinstance(active_sdr, SDR))
        self.input_sdr            = input_sdr
        self.active_sdr           = active_sdr
        self.segments_per_cell    = int(round(segments_per_cell))
        self.synapses_per_segment = int(round(synapses_per_segment))
        self.add_synapses         = int(round(add_synapses))
        self.initial_segment_size = int(round(initial_segment_size))
        self.predictive_threshold = predictive_threshold
        self.learning_threshold   = learning_threshold

        self.synapses = SynapseManager(
            input_sdr         = input_sdr,
            output_sdr        = SDR((self.active_sdr.size, self.segments_per_cell)),
            permanence_thresh = permanence_thresh,
            permanence_inc    = permanence_inc,
            permanence_dec    = permanence_dec,)
        self.mispredict_dec   = mispredict_dec

    def compute(self, input_sdr=None):
        self.excitement          = self.synapses.compute(input_sdr=input_sdr)
        self.predictive_segments = self.excitement >= self.predictive_threshold
        self.predictions         = np.sum(self.predictive_segments, axis=1)
        return self.predictions.reshape(self.active_sdr.dimensions)

    def learn(self, active_sdr=None,):
        """!"""
        self.active_sdr.assign(active_sdr)
        self._mispredict()
        self._reinforce()
        self._add_synapses()   # Call reinforce before add_synapses!

    def _mispredict(self):
        """
        All mispredicted segments receive a small permanence penalty (predictive
        segments in inactive neurons).  This penalizes only the synapses with
        active presynapses.
        """
        mispredictions = np.array(self.predictive_segments, copy=True)
        mispredictions[self.active_sdr.flat_index] = 0
        self.synapses.learn_outputs(output_sdr     = mispredictions,
                                    permanence_inc = -self.mispredict_dec,
                                    permanence_dec = 0,)

    def _reinforce(self):
        """
        Reinforce segments which correctly predicted their neurons activation.
        Allows all segments which meet the learning threshold to learn.
        """
        self.learning_segments = self.excitement[self.active_sdr.flat_index] >= self.learning_threshold
        active_num, seg_idx    = np.nonzero(self.learning_segments)
        active_idx             = self.active_sdr.flat_index[active_num]
        self.synapses.learn_outputs(output_sdr = (active_idx, seg_idx),)

    def _add_synapses(self):
        # Add more synapses to learning segments in neurons which are both
        # active and unpredicted.
        unpred_active_mask = np.logical_not(self.predictions[self.active_sdr.flat_index])
        neur_num, seg_idx  = np.nonzero(self.learning_segments[unpred_active_mask])
        neur_idx           = self.active_sdr.flat_index[neur_num]
        self.synapses.add_synapses(
            output_sdr          = (neur_idx, seg_idx),
            synapses_per_output = self.add_synapses,
            maximum_synapses    = self.synapses_per_segment)

        # Start new segments on active neurons with no learning segments. Use
        # the segments with the fewest existing synapses, which improves the
        # amount of room for the new segment to grow as well as evenly
        # distributes synapses amongst the segments.
        need_more_segments_mask = np.logical_not(np.any(self.learning_segments, axis=1))
        neur_idx                = self.active_sdr.flat_index[need_more_segments_mask]
        if len(neur_idx):
            segment_sources     = self.synapses.postsynaptic_sources.reshape(self.synapses.outputs.dimensions)
            segment_sizes       = np.array([[seg.shape[0] for seg in seg_slice]
                                    for seg_slice in segment_sources[neur_idx, :]])
            seg_idx             = np.argmin(segment_sizes, axis=1)
            self.synapses.add_synapses(
                    output_sdr          = (neur_idx, seg_idx),
                    synapses_per_output = self.initial_segment_size,
                    maximum_synapses    = self.synapses_per_segment)

    def copy(self):
        cpy            = copy.copy(self)
        cpy.synapses   = self.synapses.copy()
        cpy.input_sdr  = cpy.synapses.inputs
        cpy.active_sdr = SDR(self.active_sdr)
        return cpy

    def statistics(self):
        stats = ''

        stats += self.synapses.statistics()

        # TODO: These should report both the individual neuron and the
        # population wide error rates.  (multiply by pop size?)
        stats += 'Theoretic False Positive Rate  {:g}\n'.format(
            float(self.synapses.inputs.false_positive_rate(
                self.initial_segment_size,
                self.predictive_threshold,
            ))
        )
        for noise in [5, 10, 20, 50]:
            stats += 'Theoretic False Negative Rate, {}% noise, {:g}\n'.format(
                noise,
                float(self.synapses.inputs.false_negative_rate(
                    noise/100,
                    self.initial_segment_size,
                    self.predictive_threshold,
                ))
            )
        return stats
