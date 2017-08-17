# cython: profile=True
"""
Library of Cython optimized routines for the module htm.py
Written by David McDougall, 2017
"""

DEF debug = False

import numpy as np
cimport numpy as np
import math
cimport cython

class SynapseManager_implicit_synapse:
    """
    Experimential, dones not currently work.

    This class is workspace for reworking the synapse lists.  I want to
    partition the sources and permanences within their arrays and the partition
    index divides the connected and disconnected synapses.
    """

    """For pyramidal neuron synapses."""
    def __init__(self, input_dimensions, output_dimensions,
        radii=None,
        potential_pool=.95,
        coincidence_inc = 0.1,          # Do not touch.
        coincidence_dec = 0.02,         # Do not touch.
        permanence_threshold = 0.5,     # Do not touch.
        diag=True):
        """
        Argument input_dimensions is tuple of input space dimensions
        Argument output_dimensions is tuple of output space dimensions
        Argument radii is tuple of convolutional radii, must be same length as output_dimensions
                 radii units are the input space units
                 radii is optional, if not given assumes no topology
        Argument potential_pool is the fraction of possible inputs to include in 
                 each columns potential pool of input sources.  Default is .95

        Arguments coincidence_inc, coincidence_dec, permanence_threshold
                In theory, permanence updates are the amount of time it takes to
                learn  a thing, with units of number of gazes.

        If output_dimensions is shorter than input_dimensions then the trailing
        input_dimensions are not convolved over, are instead broadcast to all
        outputs which are connected via the convolution in the other dimensions.
        """
        self.input_dimensions     = tuple(input_dimensions)
        self.output_dimensions    = tuple(output_dimensions)
        self.num_outputs          = np.product(self.output_dimensions)
        self.coincidence_inc      = coincidence_inc
        self.coincidence_dec      = coincidence_dec
        self.permanence_threshold = permanence_threshold

        if diag:
            if isinstance(diag, str):
                print(diag, "Synapse Parameters")
            else:
                print("Synapse Parameters")
            print("\tInput -> Output shapes are",
                            self.input_dimensions, '->', self.output_dimensions)

            # TODO: Multiply both sides of the coincidence ratio by their
            # greatest common demonimator and then have both sides of the ratio
            # labeled as "Presynaptic coincidence threshold:  X Active : Y
            # Silent" And then view the scale differently. What I should really
            # do is use the ratio and scale as parameters and calculate the inc
            # and dec.
            coincidence_ratio = self.coincidence_inc / self.coincidence_dec
            print('\tCoincidence Ratio', self.coincidence_inc, '/',
                        self.coincidence_dec, '=', coincidence_ratio)

        # Both inputs and outputs are identified by their flat-index, which is
        # their index into their space after it's been flattened.  All outputs
        # have the same number of inputs in their potential pool.

        # self.sources[output-index][input-number] = input-index
        # self.sources.shape = (num_outputs, num_inputs)
        if radii is not None:
            self.radii = radii
            self.normally_distributed_connections(input_dimensions, output_dimensions, potential_pool, radii)
            if diag:
                print("\tDensity within 1/2/3 deviations: %.3g / %.3g / %.3g"%(
                            self.potential_pool_density_1,
                            self.potential_pool_density_2,
                            self.potential_pool_density_3,))
        else:
            self.dense_connections(input_dimensions, output_dimensions, potential_pool)

        self.num_inputs  = self.sources.shape[1]
        # Random permanence and synapse initialization
        self.permanences = np.random.random(self.sources.shape)
        # Synapse partition is index of first connected input in the sorted
        # sources list.  This is recomputed by self.reset()
        self.synapse_partitions = np.zeros(self.num_outputs, dtype=np.int)
        self.reset()

        if diag:
            # TODO: Synapse Manager's init diag should print the potential pool fraction
            # I'd like to to print: "XXX / YYYY"
            #   where XXX is the size of the potential pool
            #     and YYYY is the maximum size of the potential pool
            #     and XXX/YYYY is the potential fraction.

            if radii is not None:
                print("\tRadii", tuple(radii), '\tNum Inputs', self.num_inputs)
            else:
                print('\tNum Inputs', self.num_inputs)

    def reset(self):
        # Recompute synapse partitions
        new_order = np.argsort(self.permanences, axis=1)
        for c in range(self.num_outputs):
            self.permanences[c] = self.permanences[c][new_order[c]]
            self.sources[c]     = self.sources[c][new_order[c]]
            connected_column_inputs = self.permanences[c] >= self.permanence_threshold
            self.synapse_partitions[c] = np.min(np.where(connected_column_inputs))

    def normally_distributed_connections(self, input_dimensions, column_dimensions, potential_pool, radii):
        """
        Connects each column to its inputs.

        Sets the attribute self.inhibition_radii which is the radii, converted
        into column space units.

        This sets the following attributes:
            potential_pool_density_1
            potential_pool_density_2
            potential_pool_density_3
        Which measure the average fraction of inputs which are potentially
        connected to each column, looking within the first three standard
        deviations of the columns receptive field.  The areas are non-
        overlapping.
        """
        assert(len(column_dimensions) == len(radii))
        radii = np.array(radii)
        input_space_size = np.product(input_dimensions)
        # Clean up the potential_pool parameter.
        if potential_pool <= 1:
            potential_pool = potential_pool * input_space_size
        potential_pool = int(round(potential_pool))

        # Split the input space into topological and extra dimensions.
        topo_dimensions  = input_dimensions[: len(column_dimensions)]
        extra_dimensions = input_dimensions[len(column_dimensions) :]

        num_columns    = int(np.product(column_dimensions))
        self.sources   = np.empty((num_columns, potential_pool), dtype=np.int)

        # Density Statistics
        self.potential_pool_density_1 = 0
        self.potential_pool_density_2 = 0
        self.potential_pool_density_3 = 0
        extra_area   = np.product(extra_dimensions)
        num_inputs_1 = extra_area * math.pi * np.product(radii)
        num_inputs_2 = extra_area * math.pi * np.product(2 * radii)
        num_inputs_3 = extra_area * math.pi * np.product(3 * radii)
        num_inputs_2 -= num_inputs_1
        num_inputs_3 -= num_inputs_1 + num_inputs_2

        # Find where the columns are in the input.  Extra input dimensions are
        # not represented here.
        column_ranges = [slice(0, size) for size in column_dimensions]
        # column_locations[input-dimension][:] = vector locations in input
        # space, one for each column.
        column_locations = [dim.flatten() for dim in np.mgrid[column_ranges]]
        padding   = radii   #
        expand_to = np.subtract(topo_dimensions, np.multiply(2, padding))
        column_spacing    = np.divide(expand_to, column_dimensions).reshape(len(topo_dimensions), 1)
        column_locations *= column_spacing
        column_locations += np.array(padding).reshape(len(topo_dimensions), 1)
        self.inhibition_radii = radii / np.squeeze(column_spacing)

        for column_index in range(num_columns):
            center = column_locations[:, column_index]
            # Make potential-pool many unique input locations.  This is an
            # iterative process: sample the normal distribution, reject
            # duplicates, repeat until done.  Working pool holds the
            # intermediate input-coordinates until it's filled and ready to be
            # spliced into self.sources[column-index, :]
            working_pool  = np.empty((0, len(input_dimensions)), dtype=np.int)
            empty_sources = potential_pool  # How many samples to take.
            for attempt in range(10):
                # Sample points from the input space and cast to valid indecies.
                # Take more samples than are needed B/C some will not be viable.
                topo_pool     = np.random.normal(center, radii, 
                                    size=(max(256, 2*empty_sources), len(radii)))
                topo_pool     = np.rint(topo_pool)   # Round towards center
                out_of_bounds = np.where(np.logical_or(topo_pool < 0, topo_pool >= topo_dimensions))
                topo_pool     = np.delete(topo_pool, out_of_bounds, axis=0)
                extra_pool = np.random.uniform(0, extra_dimensions, size=(topo_pool.shape[0], len(extra_dimensions)))
                extra_pool = np.floor(extra_pool) # Round down to stay in half open range [0, dim)
                # Combine topo & extra dimensions into input space coordinates.
                pool       = np.concatenate([topo_pool, extra_pool], axis=1)
                pool       = np.array(pool, dtype=np.int)
                # Add the points to the working pool
                working_pool = np.concatenate([working_pool, pool], axis=0)
                # Reject duplicates
                working_pool  = np.unique(working_pool, axis=0)
                empty_sources = potential_pool - working_pool.shape[0]
                if empty_sources <= 0:
                    break
            else:
                if empty_sources > .05 * potential_pool:
                    raise ValueError("Not enough sources to fill potential pool.")
                else:
                    print("Warning: Could not find enough unique inputs, allowing %d duplicates..."%empty_sources)
                    duplicates = np.random.randint(0, working_pool.shape[0], size=empty_sources)
                    duplicates = working_pool[duplicates]
                    working_pool = np.concatenate([working_pool, duplicates], axis=0)
            working_pool = working_pool[:potential_pool, :] # Discard extra samples

            # Measure some statistics about input density.
            displacements = working_pool[:, :len(topo_dimensions)] - center
            # Measure in terms of standard deviations of their distribution.
            deviations = displacements / radii
            distances  = np.sum(deviations**2, axis=1)**.5
            pp_size_1  = np.count_nonzero(distances <= 1)
            pp_size_2  = np.count_nonzero(np.logical_and(distances > 1, distances <= 2))
            pp_size_3  = np.count_nonzero(np.logical_and(distances > 2, distances <= 3))
            self.potential_pool_density_1 += pp_size_1 / num_inputs_1
            self.potential_pool_density_2 += pp_size_2 / num_inputs_2
            self.potential_pool_density_3 += pp_size_3 / num_inputs_3

            # Flatten and write to output array.
            working_pool = np.ravel_multi_index(working_pool.T, input_dimensions)
            self.sources[column_index, :] = working_pool
        self.potential_pool_density_1 /= num_columns
        self.potential_pool_density_2 /= num_columns
        self.potential_pool_density_3 /= num_columns

    def dense_connections(self, input_dimensions, output_dimensions, potential_pool):
        """
        Connect every potential_pool inputs to every output.
        Directly sets the sources array, no returned value.
        """
        input_space_size = np.product(input_dimensions)
        input_space      = range(input_space_size)
        if potential_pool <= 1:
            potential_pool = potential_pool * input_space_size
        potential_pool = int(round(potential_pool))
        potential_pool = min(input_space_size, potential_pool)
        num_outputs  = int(np.product(output_dimensions))
        self.sources = np.empty((num_outputs, potential_pool), dtype=np.int)
        for output in range(num_outputs):
            self.sources[output] = np.random.choice(input_space, potential_pool, replace=False)

    @cython.boundscheck(debug) # turn off bounds-checking for entire function
    @cython.wraparound(debug)  # turn off negative index wrapping for entire function
    def compute(self, input_activity):
        """
        This uses the given presynaptic activity to determine the postsynaptic
        excitment.

        Returns the excitement as a flattened array.  
                Reshape to output_dimensions if needed.
        """
        if isinstance(input_activity, tuple) or input_activity.shape != self.input_dimensions:
            # It's significantly faster to make sparse inputs dense than to use
            # np.in1d, especially since this does NOT discard inactive columns.
            dense = np.zeros(self.input_dimensions, dtype=np.bool)
            dense[input_activity] = True
            input_activity = dense
        assert(input_activity.dtype == np.bool) # Otherwise self.learn->np.choose breaks
        self.input_activity = input_activity.reshape(-1)

        # Gather the inputs and sum for excitments.
        cdef np.ndarray[np.int8_t, ndim=1] inp       = np.array(self.input_activity, dtype=np.int8)
        cdef np.ndarray[np.int_t,  ndim=2] sources   = self.sources
        cdef np.ndarray[np.int_t,  ndim=1] synapses  = self.synapse_partitions
        cdef np.ndarray[np.int_t,  ndim=1] excitment = np.zeros(self.num_outputs, dtype=np.int)

        cdef int c, part, i, src, pp
        pp = self.num_inputs
        for c in range(self.num_outputs):
            part = synapses[c]
            for i in range(part, pp):
                src = sources[c, i]
                excitment[c] += inp[src]

        return excitment
        return excitment.reshape(self.output_dimensions)

    @cython.boundscheck(debug) # turn off bounds-checking for entire function
    @cython.wraparound(debug)  # turn off negative index wrapping for entire function
    def learn(self, output_activity):
        """
        Update permanences and then synapses.

        Argument output_activity is index array
        """
        cdef np.ndarray[np.int8_t, ndim=1] inp       = np.array(self.input_activity, dtype=np.int8)
        cdef np.ndarray[np.int_t,  ndim=1] out       = output_activity
        cdef np.ndarray[np.int_t,  ndim=2] sources   = self.sources
        cdef np.ndarray[np.float_t,  ndim=2] perms   = self.permanences
        cdef np.ndarray[np.int_t,  ndim=1] synapses  = self.synapse_partitions
        cdef int pp = self.num_inputs
        cdef int c, i, src, part
        cdef bint active_synapse
        cdef float p
        cdef float inc    = self.coincidence_inc
        cdef float dec    = self.coincidence_dec
        cdef float thresh = self.permanence_threshold
        for c in out:
            for i in range(pp):
                src = sources[c, i]
                active_synapse = inp[src]
                p = perms[c, i]
                if active_synapse == 0:
                    p = min(1, p - dec)
                else:
                    p = max(0, p + inc)
                perms[c, i] = p

            # Make a second pass to partition the synapses
            part = synapses[c]
            for i in range(part):
                if perms[c, i] >= thresh:
                    # Swap synapses [c,i] with [c, part-1] and move the partition
                    # across/over it.
                    p = perms[c, i]
                    perms[c, i] = perms[c, part-1]
                    perms[c, part-1] = p
                    src = sources[c, i]
                    sources[c, i] = sources[c, part-1]
                    sources[c, part-1] = src
                    part -= 1   # C for loop semantics :)
                    i -= 1      # Reprocess the current element as it has changed.

            for i in range(part, pp):
                if perms[c, i] < thresh:
                    # Swap synapses [c,i] with [c, part] and move the partition
                    # across/over it.
                    p = perms[c, i]
                    perms[c, i] = perms[c, part]
                    perms[c, part] = p
                    src = sources[c, i]
                    sources[c, i] = sources[c, part]
                    sources[c, part] = src
                    part += 1
            synapses[c] = part

