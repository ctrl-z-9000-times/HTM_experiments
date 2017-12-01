# Written by David McDougall, 2017
"""
BUILD COMMAND
./setup.py build_ext --inplace
"""

import numpy as np
import scipy.ndimage
import random
import copy
from genetics import Parameters
from encoders import *
from classifiers import *
from sdr import SDR, SynapseManager, Dendrite


class SpatialPoolerParameters(Parameters):
    parameters = [
        "permanence_inc",
        "permanence_dec",
        "permanence_thresh",
        "sparsity",
        "potential_pool",
        "boosting_alpha",
        # "init_dist",
    ]
    def __init__(self,
        permanence_inc     = 0.04,
        permanence_dec     = 0.01,
        permanence_thresh   = 0.4,
        potential_pool      = 2048,
        sparsity            = 0.02,
        boosting_alpha      = 0.001,
        # init_dist           = (0.4/4, 0.4/3),
        ):
        """
        This class contains the global parameters, which are invariant between
        different cortical regions.  The column dimensions and radii are stored
        elsewhere.

        Argument boosting_alpha is the small constant used by the moving 
                 exponential average which tracks each columns activation 
                 frequency.
        """
        # Get the parent class to save all these parameters as attributes of the same name.
        kw_args = locals()
        del kw_args['self']
        super().__init__(**kw_args)

class SpatialPooler:
    """
    This class handles the mini-column structures and the feed forward 
    proximal inputs to each cortical mini-column.

    This implementation is based on but differs from the one described by
    Numenta's Spatial Pooler white paper, (Cui, Ahmad, Hawkins, 2017, "The HTM
    Spatial Pooler - a neocortical...") in two main ways, the boosting function
    and the local inhibition mechanism.

    Logarithmic Boosting Function:
    This uses a logarithmic boosting function.  Its input is the activation
    frequency which is in the range [0, 1] and its output is a boosting factor
    to multiply each columns excitement by.  It's equation is:
        boost-factor = log( activation-frequency ) / log( target-frequency )
    Some things to note:
        1) The boost factor asymptotically approaches infinity as the activation
           frequency approaches zero.
        2) The boost factor equals zero when the actiavtion frequency is one.
        3) The boost factor for columns which are at the target activation 
           frequency is one.  
        4) This mechanism has a single parameter: boosting_alpha which controls
           the exponential moving average which tracks the activation frequency.

    Fast Local Inhibition:
    This activates the most excited columns globally, after normalizing all
    columns by their local area mean and standard deviation.  The local area is
    a gaussian window and the standard deviation of it is proportional to the
    deviation which is used to make the receptive fields of each column.
    Columns inhibit each other in proportion to the number of inputs which they
    share.  In pseudo code:
    1.  mean_normalized = excitement - gaussian_blur( excitement, radius )
    2.  standard_deviation = sqrt( gaussian_blur( mean_normalized ^ 2, radius ))
    3.  normalized = mean_normalized / standard_deviation
    4.  activate = top_k( normalized, sparsity * number_of_columns )
    """
    stability_st_period = 1000
    stability_lt_period = 10       # Units: self.stability_st_period

    def __init__(self, parameters, input_sdr, column_sdr,
        radii=None,
        stability_sample_size=0,
        multisegment_experiment=None,
        init_dist=None,):
        """
        Argument parameters is an instance of SpatialPoolerParameters.

        Argument input_sdr ...
        Argument column_sdr ...

        Argument radii is the standard deviation of the gaussian window which
                 defines the local neighborhood of a column.  The radii
                 determine which inputs are likely to be in a columns potential
                 pool.  If radii is None then topology is disabled.  See
                 SynapseManager.normally_distributed_connections for details
                 about topology.

        Argument stability_sample_size, set to 0 to disable stability
                 monitoring, default is off.  
        """
        assert(isinstance(parameters, SpatialPoolerParameters))
        assert(isinstance(input_sdr, SDR))
        assert(isinstance(column_sdr, SDR))
        self.args = args           = parameters
        self.inputs                = input_sdr
        self.columns               = column_sdr
        self.topology              = radii is not None
        self.age                   = 0
        self.stability_schedule    = [0] if stability_sample_size > 0 else [-1]
        self.stability_sample_size = stability_sample_size
        self.stability_samples     = []

        self.multisegment = multisegment_experiment is not None
        if self.multisegment:
            # EXPERIMENTIAL: Multi-segment proximal dendrites.
            self.segments_per_cell = int(round(multisegment_experiment))
            self.proximal = SynapseManager( self.inputs,
                                            SDR(self.columns.dimensions + (self.segments_per_cell,),
                                                activation_frequency_alpha=args.boosting_alpha),    # Used for boosting!
                                            permanence_inc    = args.permanence_inc,
                                            permanence_dec    = args.permanence_dec,
                                            permanence_thresh = args.permanence_thresh,)
            # Initialize to the target activation frequency/sparsity.
            self.proximal.outputs.activation_frequency.fill(args.sparsity / self.segments_per_cell)
        else:
            self.proximal = SynapseManager( self.inputs,
                                            self.columns,
                                            permanence_inc    = args.permanence_inc,
                                            permanence_dec    = args.permanence_dec,
                                            permanence_thresh = args.permanence_thresh,)
        if self.topology:
            r = self.proximal.normally_distributed_connections(args.potential_pool, radii, init_dist=init_dist)
            self.inhibition_radii = r
        else:
            self.proximal.uniformly_distributed_connections(args.potential_pool, init_dist=init_dist)

        if args.boosting_alpha is not None:
            # Make a dedicated SDR to track column activation frequencies for
            # boosting.
            self.boosting = SDR(self.columns,
                                activation_frequency_alpha = args.boosting_alpha,
                                # Note: average overlap is useful to know, but is not part of the boosting algorithm.
                                average_overlap_alpha      = args.boosting_alpha,)
            # Initialize to the target activation frequency/sparsity.
            self.boosting.activation_frequency.fill(args.sparsity)

    def compute(self, input_sdr=None):
        """
        """
        args = self.args
        if self.multisegment:
            # EXPERIMENT: Multi segment proximal dendrites.
            excitment = self.proximal.compute(input_sdr=input_sdr)

            # Logarithmic Boosting Function.
            if args.boosting_alpha is not None:
                target_sparsity = args.sparsity / self.segments_per_cell
                boost = np.log2(self.proximal.outputs.activation_frequency) / np.log2(target_sparsity)
                boost = np.nan_to_num(boost).reshape(self.proximal.outputs.dimensions)
                excitment = boost * excitment

            # Break ties randomly
            excitment = excitment + np.random.uniform(0, .5, size=self.proximal.outputs.dimensions)

            self.segment_excitement = excitment
            # Replace the segment dimension with each columns most excited segment.
            excitment = np.max(excitment, axis=-1)
            raw_excitment = excitment.reshape(-1)
        else:
            raw_excitment = self.proximal.compute(input_sdr=input_sdr).reshape(-1)

            # Logarithmic Boosting Function.
            if args.boosting_alpha is not None:
                boost = np.log2(self.boosting.activation_frequency) / np.log2(args.sparsity)
                boost = np.nan_to_num(boost)
                raw_excitment = boost * raw_excitment

        # Fast Local Inhibition
        if self.topology:
            inhibition_radii    = self.inhibition_radii
            raw_excitment       = raw_excitment.reshape(self.columns.dimensions)
            avg_local_excitment = scipy.ndimage.filters.gaussian_filter(
                                    # Truncate for speed
                                    raw_excitment, inhibition_radii, mode='reflect', truncate=3.0)
            local_excitment     = raw_excitment - avg_local_excitment
            stddev              = np.sqrt(scipy.ndimage.filters.gaussian_filter(
                                    local_excitment**2, inhibition_radii, mode='reflect', truncate=3.0))
            raw_excitment       = np.nan_to_num(local_excitment / stddev)
            raw_excitment       = raw_excitment.reshape(-1)

        # EXPERIMENTIAL
        self.raw_excitment = raw_excitment

        # Activate the most excited columns.
        #
        # Note: excitements are not normally distributed, their local
        # neighborhoods use gaussian windows, which are a different thing. Don't
        # try to use a threshold, it won't work.  Especially not: threshold =
        # scipy.stats.norm.ppf(1 - sparsity).
        k = self.columns.size * args.sparsity
        k = max(1, int(round(k)))
        self.columns.flat_index = np.argpartition(-raw_excitment, k-1)[:k]
        return self.columns

    def learn(self, input_sdr=None, column_sdr=None):
        """
        Make the spatial pooler learn about its current inputs and active columns.
        """
        if self.multisegment:
            # Learn about regular activations
            self.columns.assign(column_sdr)
            segment_excitement = self.segment_excitement[self.columns.index]
            seg_idx = np.argmax(segment_excitement, axis=-1)
            # seg_idx = np.random.choice(self.segments_per_cell, size=len(self.columns))
            self.proximal.learn_outputs(input_sdr=input_sdr,
                                        output_sdr=self.columns.index + (seg_idx,))
        else:
            # Update proximal synapses and their permanences.  Also assigns into our column SDR.
            self.proximal.learn_outputs(input_sdr=input_sdr, output_sdr=column_sdr)
        # Update the exponential moving average of each columns activation frequency.
        self.boosting.assign(self.columns)
        # Book keeping.
        self.stability(self.inputs, self.columns.index)
        self.age += 1

    def stabilize(self, prior_columns, percent):
        """
        This activates prior columns to force active in order to maintain
        the given percent of column overlap between time steps.  Always call
        this between compute and learn!
        """
        # num_active      = (len(self.columns) + len(prior_columns)) / 2
        num_active      = len(self.columns)
        overlap         = self.columns.overlap(prior_columns)
        stabile_columns = int(round(num_active * overlap))
        target_columns  = int(round(num_active * percent))
        add_columns     = target_columns - stabile_columns
        if add_columns <= 0:
            return

        eligable_columns  = np.setdiff1d(prior_columns.flat_index, self.columns.flat_index)
        eligable_excite   = self.raw_excitment[eligable_columns]
        selected_col_nums = np.argpartition(-eligable_excite, add_columns-1)[:add_columns]
        selected_columns  = eligable_columns[selected_col_nums]
        selected_index    = np.unravel_index(selected_columns, self.columns.dimensions)
        # Learn.  Note: selected columns will learn twice.  The previously
        # active segments learn now, the current most excited segments in the
        # method SP.learn().
        # Or learn not at all if theres a bug in my code...
        # if self.multisegment:
        #     if hasattr(self, 'prior_segment_excitement'):
        #         segment_excitement = self.prior_segment_excitement[selected_index]
        #         seg_idx = np.argmax(segment_excitement, axis=-1)
        #         self.proximal.learn_outputs(input_sdr=input_sdr,
        #                                     output_sdr=selected_index + (seg_idx,))
        #     self.prev_segment_excitement = self.segment_excitement
        # else:
        #     1/0
        self.columns.flat_index = np.concatenate([self.columns.flat_index, selected_columns])

    def plot_boost_functions(self, beta = 15):
        # Generate sample points
        dc = np.linspace(0, 1, 10000)
        from matplotlib import pyplot as plt
        fig = plt.figure(1)
        ax = plt.subplot(111)
        log_boost = lambda f: np.log(f) / np.log(self.args.sparsity)
        exp_boost = lambda f: np.exp(beta * (self.args.sparsity - f))
        logs = [log_boost(f) for f in dc]
        exps = [exp_boost(f) for f in dc]
        plt.plot(dc, logs, 'r', dc, exps, 'b')
        plt.title("Boosting Function Comparison \nLogarithmic in Red, Exponential in Blue (beta = %g)"%beta)
        ax.set_xlabel("Activation Frequency")
        ax.set_ylabel("Boost Factor")
        plt.show()

    def stability(self, input_sdr, output_sdr, diag=True):
        """
        Measures the short and long term stability from compute's input stream.
        Do not call this directly!  Instead set it up before and via 
        SpatialPooler.__init__() and this will print the results to STDOUT.

        Argument input_sdr, output_sdr ...

        Attribute stability_sample_size is how many samples to take during each
                  sample period.  

        Attribute stability_samples is list of samples, where each sample is a 
                  list of pairs of (input_sdr, output_sdr).  The index is how 
                  many (short term) sample periods ago the sample was taken.

        Attribute stability_schedule is a list of ages to take input/output
                  samples at, in descending order so that the soonest sample age
                  is at the end of the list.   Append -1 to the schedule to
                  disable stability monitoring. The final age in the schedule is
                  special, on this age it calculates the stability and makes a
                  new schedule for the next period.

        Class Attribute stability_st_period
                st == short term, lt == long term
                The stability period is how many compute cycles this SP will
                wait before recomputing the stability samples and comparing with
                the original results. This calculates two measures of stability:
                short and long term.  The long  term period is written in terms
                of the short term period.

        Class Attribute stability_lt_period
                    Units: self.stability_st_period

        Attribute st_stability, lt_stability are the most recent measurements of
                  short and long term stability, respectively.  These are 
                  initialized to None.
        """
        if self.stability_schedule[-1] != self.age:
            return
        else:
            self.stability_schedule.pop()

        if self.stability_schedule:
            # Not the final scheduled checkup. Take the given sample and return.  
            self.stability_samples[0].append((input_sdr, output_sdr))
            return
        # Else: calculate the stability and setup for the next period of 
        # stability sampling & monitoring.  

        assert(False) # This method probably won't work since changes to use SDR class...

        def overlap(a, b):
            a = set(zip(*a))
            b = set(zip(*b))
            overlap = len(a.intersection(b))
            overlap_pct = overlap / min(len(a), len(b))
            return overlap_pct

        # Rerun the samples through the machine.  
        try:
            st_samples = self.stability_samples[1]
        except IndexError:
            self.st_stability = None    # This happens when age < 2 x st_period
        else:
            st_rerun = [self.compute(inp, learn=False) for inp, out in st_samples]
            self.st_stability = np.mean([overlap(re, io[1]) for re, io in zip(st_rerun, st_samples)])

        try:
            lt_samples = self.stability_samples[self.stability_lt_period]
        except IndexError:
            self.lt_stability = None    # This happens when age < st_period X (lt_period + 1)
        else:
            lt_rerun   = [self.compute(inp, learn=False) for inp, out in lt_samples]
            self.lt_stability = np.mean([overlap(re, io[1]) for re, io in zip(lt_rerun, lt_samples)])

        # Make a new sampling schedule.
        sample_period = range(self.age + 1, self.age + self.stability_st_period)
        self.stability_schedule = random.sample(sample_period, self.stability_sample_size)
        # Add the next stability calculation to the end of the schedule.  
        self.stability_schedule.append(sample_period.stop)
        self.stability_schedule.sort(reverse=True)
        # Roll the samples buffer.
        self.stability_samples.insert(0, [])
        self.stability_samples = self.stability_samples[:self.stability_lt_period + 1]

        # Print output
        if diag:
            s = ""
            if self.st_stability is not None:
                s += "Stability (%d) %-5.03g"%(self.stability_st_period, self.st_stability,)
            if self.lt_stability is not None:
                s += " | (x%d) %-5.03g"%(self.stability_lt_period, self.lt_stability)
            if s:
                print(s)

    def noise_perturbation(self, inp, flip_bits, diag=False):
        """
        Measure the change in SDR overlap after moving some of the ON bits.
        """
        tru = self.compute(inp, learn=False)

        # Make sparse input dense.
        if isinstance(inp, tuple) or inp.shape != self.args.input_dimensions:
            dense = np.zeros(self.args.input_dimensions)
            dense[inp] = True
            inp = dense

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
            cutoff = len(noises) // 10          # First 'cutoff' many samples have full accuracy.
            while len(noises) > 50 + cutoff:    # Decimate to a sane number of sample points
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
            plt.title('todo')
            plt.xlabel('todo')
            plt.ylabel('todo')
            plt.show()

        return noises, pct_over

    def statistics(self):
        stats = 'SP '
        stats += self.proximal.statistics()

        if self.args.boosting_alpha is not None:
            stats      += 'Columns ' + self.boosting.statistics()
            af         = self.boosting.activation_frequency
            boost_min  = np.log2(np.min(af))  / np.log2(self.args.sparsity)
            boost_mean = np.log2(np.mean(af)) / np.log2(self.args.sparsity)
            boost_max  = np.log2(np.max(af))  / np.log2(self.args.sparsity)
            stats += '\tLogarithmic Boosting Multiplier min/mean/max  {:-.04g}% / {:-.04g}% / {:-.04g}%\n'.format(
                    boost_min   * 100,
                    boost_mean  * 100,
                    boost_max   * 100,)

        # TODO: Stability, if enabled.
        pass

        # TODO: Noise robustness, if enabled.
        pass

        return stats


class TemporalMemoryParameters(Parameters):
    parameters = [
        'add_synapses',             # How many new synapses to add to subthreshold learning segments.
        'cells_per_column',
        'initial_segment_size',     # How many synases to start new segments with.
        'segments_per_cell',
        'synapses_per_segment',
        'permanence_inc',
        'permanence_dec',
        'mispredict_dec',
        'permanence_thresh',
        'predictive_threshold',     # Segment excitement threshold for predictions.
        'learning_threshold',       # Segment excitement threshold for learning.
    ]
    def __init__(self,
        cells_per_column        = 1.022e+01,
        learning_threshold      = 7.215e+00,
        mispredict_dec          = 1.051e-03,
        permanence_dec          = 9.104e-03,
        permanence_inc          = 2.272e-02,
        permanence_thresh       = 2.708e-01,
        predictive_threshold    = 6.932e+00,
        segments_per_cell       = 1.404e+02,
        synapses_per_segment    = 1.190e+02,
        add_synapses            = 1,
        initial_segment_size    = 10,
        ):
        # Get the parent class to save all these parameters as attributes of the same name.
        super().__init__(**{k:v for k,v in locals().items() if k != 'self'})

class TemporalMemory:
    """
    This implementation is based on the paper: Hawkins J. and Ahmad S. (2016)
    Why Neurons Have Thousands of Synapses, a Theory of Sequency Memory in
    Neocortex. Frontiers in Neural Circuits 10:23 doi: 10.3389/fncir.2016.00023
    """
    def __init__(self, 
        parameters,
        column_sdr,
        apical_sdr=None,
        inhibition_sdr=None,
        context_sdr=None,
        ):
        """
        Argument parameters is an instance of TemporalMemoryParameters
        Argument column_dimensions ...
        """
        assert(isinstance(parameters, TemporalMemoryParameters))
        self.args = args         = parameters
        assert(isinstance(column_sdr, SDR))
        self.columns             = column_sdr
        self.cells_per_column    = int(round(args.cells_per_column))
        if self.cells_per_column < 1:
            raise ValueError("Cannot create TemporalMemory with cells_per_column < 1.")
        self.segments_per_cell   = int(round(args.segments_per_cell))
        self.active              = SDR((self.columns.size, self.cells_per_column),
                                        activation_frequency_alpha = 1/1000,
                                        average_overlap_alpha      = 1/1000,)
        self.anomaly_alpha       = 1/1000
        self.mean_anomaly        = 0

        self.basal = Dendrite(
            input_sdr            = SDR(context_sdr if context_sdr is not None else self.active),
            active_sdr           = SDR(self.active),
            segments_per_cell    = args.segments_per_cell,
            synapses_per_segment = args.synapses_per_segment,
            initial_segment_size = args.initial_segment_size,
            add_synapses         = args.add_synapses,
            learning_threshold   = args.learning_threshold,
            predictive_threshold = args.predictive_threshold,
            permanence_inc       = args.permanence_inc,
            permanence_dec       = args.permanence_dec,
            permanence_thresh    = args.permanence_thresh,
            mispredict_dec       = args.mispredict_dec,)

        if apical_sdr is None:
            self.apical = None
        else:
            assert(isinstance(apical_sdr, SDR))
            self.apical = Dendrite(
                input_sdr            = apical_sdr,
                active_sdr           = self.active,
                segments_per_cell    = args.segments_per_cell,
                synapses_per_segment = args.synapses_per_segment,
                initial_segment_size = args.initial_segment_size,
                add_synapses         = args.add_synapses,
                learning_threshold   = args.learning_threshold,
                predictive_threshold = args.predictive_threshold,
                permanence_inc       = args.permanence_inc,
                permanence_dec       = args.permanence_dec,
                permanence_thresh    = args.permanence_thresh,
                mispredict_dec       = args.mispredict_dec,)

        if inhibition_sdr is None:
            self.inhibition = None
        else:
            assert(isinstance(inhibition_sdr, SDR))
            self.inhibition = Dendrite(
                input_sdr            = inhibition_sdr,
                active_sdr           = self.active,
                segments_per_cell    = args.segments_per_cell,
                synapses_per_segment = args.synapses_per_segment,
                initial_segment_size = args.initial_segment_size,
                add_synapses         = args.add_synapses,
                learning_threshold   = args.learning_threshold,
                predictive_threshold = args.predictive_threshold,
                permanence_inc       = args.permanence_inc,
                permanence_dec       = args.permanence_dec,
                permanence_thresh    = args.permanence_thresh,
                mispredict_dec       = 0,) # Is not but should be an inhibited segment in an active cell.

        self.reset()

    def reset(self):
        self.active.zero()
        self.reset_state = True

    def compute(self,
        context_sdr=None,
        column_sdr=None,
        apical_sdr=None,
        inhibition_sdr=None,):
        """
        Attribute anomaly, mean_anomaly are the fraction of neuron activations
                  which were predicted.  Range [0, 1]
        """
        ########################################################################
        # PHASE 1:  Make predictions based on the previous timestep.
        ########################################################################
        if context_sdr is None:
            context_sdr = self.active
        basal_predictions = self.basal.compute(input_sdr=context_sdr)
        predictions       = basal_predictions

        if self.apical is not None:
            apical_predictions = self.apical.compute(input_sdr=apical_sdr)
            predictions        = np.logical_or(predictions, apical_predictions)

        # Inhibition cancels out predictions.  The technical term is
        # hyper-polarization.  Practically speaking, this is needed so that
        # inhibiting neurons can cause mini-columns to burst.
        if self.inhibition is not None:
            inhibited   = self.inhibition.compute(input_sdr=inhibition_sdr)
            predictions = np.logical_and(predictions, np.logical_not(inhibited))

        ########################################################################
        # PHASE 2:  Determine the currently active neurons.
        ########################################################################
        self.columns.assign(column_sdr)
        columns = self.columns.flat_index

        # Activate all neurons which are in a predictive state and in an active
        # column, unless they are inhibited by apical input.
        active_dense      = predictions[columns]
        col_num, neur_idx = np.nonzero(active_dense)
        # This gets the actual column index, undoes the effect of discarding the
        # inactive columns before the nonzero operation.  
        col_idx           = columns[col_num]
        predicted_active  = (col_idx, neur_idx)

        # If a column activates but was not predicted by any neuron segment,
        # then it bursts.  The bursting columns are the unpredicted columns.
        bursting_columns = np.setdiff1d(columns, col_idx)
        # All neurons in bursting columns activate.
        burst_col_idx  = np.repeat(bursting_columns, self.cells_per_column)
        burst_neur_idx = np.tile(np.arange(self.cells_per_column), len(bursting_columns))
        burst_active   = (burst_col_idx, burst_neur_idx)
        # Apply inhibition to the bursting mini-columns.
        if self.inhibition is not None:
            uninhibited_mask = np.logical_not(inhibited[burst_active])
            burst_active     = np.compress(uninhibited_mask, burst_active, axis=1)

        # TODO: Combined apical and basal predictions can cause L5 cells to
        # spontaneously activate.
        if False:
            volunteers = np.logical_and(self.basal_predictions, self.apical_predictions)
            volunteers = np.nonzero(volunteers.ravel())
            unique1d(volunteers, predicted_active+burst_active)

        self.active.index = tuple(np.concatenate([predicted_active, burst_active], axis=1))

        # Only tell the dendrite about active cells which are allowed to learn.
        bursting_learning = (
            bursting_columns,
            np.random.randint(0, self.cells_per_column, size=len(bursting_columns)))
        # TODO: This will NOT work for CONTEXT, TM ONLY.
        self.basal.input_sdr.assign(self.basal.active_sdr) # Only learn about the winner cells from last cycle.
        self.basal.active_sdr.index = tuple(np.concatenate([predicted_active, bursting_learning], axis=1))

        # Anomally metric.
        self.anomaly      = np.array(burst_active).shape[1] / len(self.active)
        alpha             = self.anomaly_alpha
        self.mean_anomaly = (1-alpha)*self.mean_anomaly + alpha*self.anomaly

    def learn(self):
        """
        Learn about the previous to current timestep transition.
        """
        if self.reset_state:
            # Learning on the first timestep after a reset is not useful. The
            # issue is that waking up after a reset is inherently unpredictable.
            self.reset_state = False
            return

        # NOTE: All cells in a bursting mini-column will learn.  This includes
        # starting new segments if necessary.  This is different from Numenta's
        # TM which choses one cell to learn on a bursting column.  If in fact
        # all newly created segments work correctly, then I may in fact be
        # destroying any chance of it learning a unique representation of the
        # anomalous sequence by assigning all cells to represent it.  I was
        # thinking that maybe this would work anyways because the presynapses
        # are chosen randomly but now its evolved an initial segment size of 19!
        # FIXED?

        # Use the SDRs which were given durring the compute phase.
        # inputs = previous winner cells, active = current winner cells
        self.basal.learn(active_sdr=None)
        if self.apical is not None:
            self.apical.learn(active_sdr=self.active)
        if self.inhibition is not None:
            self.inhibition.learn(active_sdr=self.active)

    def statistics(self):
        stats  = 'Temporal Memory\n'
        stats += 'Predictive Segments ' + self.basal.statistics()
        if self.apical is not None:
            stats += 'Apical Segments ' + self.apical.statistics()
        if self.inhibition is not None:
            stats += 'Inhibition Segments ' + self.inhibition.statistics()

        stats += "Mean anomaly %g\n"%self.mean_anomaly
        stats += 'Activation statistics ' + self.active.statistics()

        return stats


class CorticalRegionParameters(Parameters):
    parameters = [
        'inp_cols',
        'inp_radii',
        'out_cols',
        'out_radii',
    ]

class CorticalRegion:
    def __init__(self, cerebrum_parameters, region_parameters,
        input_sdr,
        context_sdr,
        apical_sdr,
        inhibition_sdr,):
        """
        Argument cerebrum_parameters is an instance of CerebrumParameters.
        Argument region_parameters is an instance of CorticalRegionParameters.
        Argument input_sdr ... feed forward
        Argument context_sdr ... all output layers, flat
        Argument apical_sdr ... from BG D1 cells
        Argument inhibition_sdr ... from BG D2 cells
        """
        assert(isinstance(cerebrum_parameters, CerebrumParameters))
        assert(isinstance(region_parameters, CorticalRegionParameters))
        self.cerebrum_parameters = cerebrum_parameters
        self.region_parameters   = region_parameters

        self.L6_sp = SpatialPooler( cerebrum_parameters.inp_sp,
                                    input_sdr  = input_sdr,
                                    column_sdr = SDR(region_parameters.inp_cols),
                                    radii      = region_parameters.inp_radii,)
        self.L6_tm = TemporalMemory(cerebrum_parameters.inp_tm,
                                    column_sdr  = self.L6_sp.columns,
                                    context_sdr = context_sdr,)

        self.L5_sp = SpatialPooler( cerebrum_parameters.out_sp,
                                    input_sdr   = self.L6_tm.active,
                                    column_sdr  = SDR(region_parameters.out_cols),
                                    radii       = region_parameters.out_radii,)
        self.L5_tm = TemporalMemory(cerebrum_parameters.out_tm,
                                    column_sdr     = self.L5_sp.columns,
                                    apical_sdr     = apical_sdr,
                                    inhibition_sdr = inhibition_sdr,)

        self.L4_sp = SpatialPooler( cerebrum_parameters.inp_sp,
                                    input_sdr   = input_sdr,
                                    column_sdr  = SDR(region_parameters.inp_cols),
                                    radii       = region_parameters.inp_radii,)
        self.L4_tm = TemporalMemory(cerebrum_parameters.inp_tm,
                                    column_sdr  = self.L4_sp.columns,
                                    context_sdr = context_sdr,)

        self.L23_sp = SpatialPooler( cerebrum_parameters.out_sp,
                                    input_sdr   = self.L4_tm.active,
                                    column_sdr  = SDR(region_parameters.out_cols),
                                    radii       = region_parameters.out_radii,)
        self.L23_tm = TemporalMemory(cerebrum_parameters.out_tm,
                                     column_sdr = self.L23_sp.columns)

    def reset(self):
        self.L6_tm.reset()
        self.L5_tm.reset()
        self.L4_tm.reset()
        self.L23_tm.reset()

    def compute(self):
        self.L6_sp.compute()
        self.L6_tm.compute()

        self.L5_sp.compute()
        self.L5_tm.compute()

        self.L4_sp.compute()
        self.L4_tm.compute()

        self.L23_sp.compute()
        self.L23_tm.compute()

    def learn(self, bg):
        self.L6_sp.learn(column_sdr=np.any(self.L6_tm.active.dense, axis=1))
        self.L6_tm.learn()
        self.L5_sp.learn(column_sdr=np.any(self.L5_tm.active.dense, axis=1))
        self.L5_tm.apical.permanence_inc     = bg.d1_inc
        self.L5_tm.apical.permanence_dec     = bg.d1_dec
        self.L5_tm.inhibition.permanence_inc = bg.d2_inc
        self.L5_tm.inhibition.permanence_dec = bg.d2_dec
        self.L5_tm.learn()
        self.L4_sp.learn(column_sdr=np.any(self.L4_tm.active.dense, axis=1))
        self.L4_tm.learn()
        self.L23_sp.learn(column_sdr=np.any(self.L23_tm.active.dense, axis=1))
        self.L23_tm.learn()

    def statistics(self):
        stats  = ''
        stats += 'L6 Proximal ' + self.L6_sp.statistics() + '\n'
        stats += 'L6 Basal '    + self.L6_tm.statistics() + '\n'
        stats += 'L5 Proximal ' + self.L5_sp.statistics() + '\n'
        stats += 'L5 Basal '    + self.L5_tm.statistics() + '\n'
        stats += 'L4 Proximal ' + self.L4_sp.statistics() + '\n'
        stats += 'L4 Basal '    + self.L4_tm.statistics() + '\n'
        stats += 'L23 Proximal ' + self.L23_sp.statistics() + '\n'
        stats += 'L23 Basal '    + self.L23_tm.statistics() + '\n'
        return stats


class CerebrumParameters(Parameters):
    parameters = [
        'alpha',
        'bg',
        'inp_sp',
        'inp_tm',
        'out_sp',
        'out_tm',
    ]

# TODO: Move motor controls into the cerebrum.  This isn't important right now
#       because I have working motor controls in the eye-experiment file.
class Cerebrum:
    """
    """
    def __init__(self, cerebrum_parameters, region_parameters, input_sdrs):
        self.cerebrum_parameters = cerebrum_parameters
        self.region_parameters   = tuple(region_parameters)
        self.inputs              = tuple(input_sdrs)
        self.age                 = 0
        assert(isinstance(cerebrum_parameters, CerebrumParameters))
        assert(all(isinstance(rgn, CorticalRegionParameters) for rgn in self.region_parameters))
        assert(len(region_parameters) == len(self.inputs))
        assert(all(isinstance(inp, SDR) for inp in self.inputs))

        # The size of the cortex needs to be known before it can be constructed.
        context_size     = 0
        self.apical_sdrs = []
        for rgn_args in self.region_parameters:
            num_cols  = np.product([int(round(dim)) for dim in rgn_args.out_cols])
            cells_per = int(round(cerebrum_parameters.out_tm.cells_per_column))
            context_size += num_cols * cells_per * 2
            L5_dims      = (num_cols * cells_per,)
            self.apical_sdrs.append((SDR(L5_dims), SDR(L5_dims)))
        self.L23_activity  = SDR((context_size/2,))
        self.L5_activity   = SDR((context_size/2,))
        self.context_sdr   = SDR((context_size,))

        # Construct the Basal Ganglia
        self.basal_ganglia = BasalGanglia(cerebrum_parameters.bg,
                                          input_sdr  = self.context_sdr,
                                          output_sdr = self.L5_activity,)

        # Construct the cortex.
        self.regions = []
        for rgn_args, inp, apical in zip(self.region_parameters, input_sdrs, self.apical_sdrs):
            rgn = CorticalRegion(cerebrum_parameters, rgn_args,
                                 input_sdr      = inp,
                                 context_sdr    = self.context_sdr,
                                 apical_sdr     = self.basal_ganglia.d1.active,
                                 inhibition_sdr = self.basal_ganglia.d2.active,)
            self.regions.append(rgn)

        # Construct the motor controls.
        pass

    def reset(self):
        self.basal_ganglia.reset()
        for rgn in self.regions:
            rgn.reset()

    def compute(self, reward, learn=True):
        """
        Runs a single cycle for a whole network of cortical regions.
        Arguments inputs and regions are parallel lists.
        Optional Argument apical_input ... dense integer array, shape=output-dimensions
        Optional argument learn ... default is True.
        """
        for rgn in self.regions:
            rgn.compute()

        self.L5_activity.assign_flat_concatenate(rgn.L5_tm.active for rgn in self.regions)
        self.L23_activity.assign_flat_concatenate(rgn.L23_tm.active for rgn in self.regions)
        self.context_sdr.assign_flat_concatenate([self.L5_activity, self.L23_activity])

        if not learn:
            reward = None
        self.basal_ganglia.compute(reward)

        if learn:
            for rgn in self.regions:
                rgn.learn(self.basal_ganglia)

        # Motor controls.
        pass

        if learn:
            self.age += 1

    def statistics(self):
        stats = ''
        for idx, rgn in enumerate(self.regions):
            stats += 'Region {}\n'.format(idx+1)
            stats += rgn.statistics() + '\n'
        # stats += self.basal_ganglia.statistics()
        return stats
