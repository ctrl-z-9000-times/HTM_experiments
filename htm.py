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
from sdr import SDR, SynapseManager


class SpatialPoolerParameters(Parameters):
    parameters = [
        "permanence_inc",
        "permanence_dec",
        "permanence_thresh",
        "sparsity",
        "potential_pool",
        "boosting_alpha",
    ]
    def __init__(self,
        permanence_inc     = 0.04,
        permanence_dec     = 0.01,
        permanence_thresh   = 0.4,
        potential_pool      = 2048,
        sparsity            = 0.02,
        boosting_alpha      = 0.001,):
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
        # Double underscores are magic and come and go as they please.  Filter them all out.
        dunder  = lambda name: name.startswith('__') and name.endswith('__')
        kw_args = {k:v for k,v in kw_args.items() if not dunder(k)}
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
    share.

    In pseudo code:
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
        multisegment_experiment=None):
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
                                                activation_frequency_alpha=args.boosting_alpha),    # DEBUG
                                            permanence_inc    = args.permanence_inc,
                                            permanence_dec    = args.permanence_dec,
                                            permanence_thresh = args.permanence_thresh,)
        else:
            self.proximal = SynapseManager( self.inputs,
                                            self.columns,
                                            permanence_inc    = args.permanence_inc,
                                            permanence_dec    = args.permanence_dec,
                                            permanence_thresh = args.permanence_thresh,)
        if self.topology:
            r = self.proximal.normally_distributed_connections(args.potential_pool, radii)
            self.inhibition_radii = r
        else:
            self.proximal.uniformly_distributed_connections(args.potential_pool)

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
            raw_excitment = self.proximal.compute(input_sdr=input_sdr)
            self.segment_excitement = raw_excitment
            # Replace the segment dimension with each columns most excited segment.
            raw_excitment = np.max(raw_excitment, axis=-1)
            raw_excitment = raw_excitment.reshape(-1)
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
            self.columns.assign(column_sdr)
            segment_excitement = self.segment_excitement[self.columns.index]
            seg_idx = np.argmax(segment_excitement, axis=-1)
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
        num_active      = (len(self.columns) + len(prior_columns)) / 2
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


class Dendrite:
    """
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
        initial_segment_size,
        ):
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
        mispredictions = np.array(self.predictive_segments)
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

    def statistics(self):
        stats = ''

        # TODO: These should report both the individual neuron and the
        # population wide error rates.  (multiply by pop size?)
        stats += 'Theoretic False Positive Rate  {:g}\n'.format(
            float(self.synapses.inputs.false_positive_rate(
                self.synapses_per_segment,
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
        kw_args = locals()
        # Double underscores are magic and come and go as they please.  Filter them all out.
        dunder  = lambda name: name.startswith('__') and name.endswith('__')
        kw_args = {k:v for k,v in kw_args.items() if not dunder(k)}
        del kw_args['self']
        super().__init__(**kw_args)

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
            input_sdr            = context_sdr if context_sdr is not None else self.active,
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
            # issue is that waking up after a reset can not be predicted by
            # short term memory so don't try to.
            self.reset_state = False
            return

        # NOTE: This will add synapses to all neurons which are unpredicted and
        # active, and it does NOT look at the other segments for predictions...
        # Either modify each dendrites internal predictions array or overhaul
        # the way synapses are added and managed.  Regardless, these extras
        # synapses shouldn't hurt...
        self.basal.learn(active_sdr=self.active)
        if self.apical is not None:
            self.apical.learn(active_sdr=self.active)
        if self.inhibition is not None:
            self.inhibition.learn(active_sdr=self.active)

    def statistics(self):
        stats = 'Temporal Memory ' + self.basal.synapses.statistics()

        stats += 'Predictive Segments ' + self.basal.statistics()
        if self.apical is not None:
            stats += 'Apical Segments ' + self.apical.statistics()
        if self.inhibition is not None:
            stats += 'Inhibition Segments ' + self.inhibition.statistics()

        stats += "Mean anomaly %g\n"%self.mean_anomaly
        stats += 'Activation statistics ' + self.active.statistics()

        return stats


"""
EXPERIMENT:  I want to check that the B.G. output can encode recognizable
actions.  Use a statistical classifier to correlate the BG output with the motor
output and then test if the BG+classifier can function.

My hypothesis is that the striatum does not encode recognisable actions.
Experiment with creating a thalamus which accepts L5 state (with topology) and
learns to control L5 via apical control (with topology).  The BG would then
control whether the thalamus is excitory or inhibitory.
"""

# TODO: Should burst cells actually activate?  It would make them participate
# in the apical control too, also learning apical control.  Currently the burst
# cells learn but are NOT put on the active list.  Also the suppressed cells are
# kept on the active list...
#
class StriatumPopulation:
    """
    This class models the D1 and the D2 populations of neurons in the striatum of
    the basal ganglia.
    """
    def __init__(self, basal_ganglia, size):
        self.args = args = basal_ganglia.args
        self.size        = size
        self.synapses    = SynapseManager(
            input_sdr           = basal_ganglia.inputs,
            output_sdr          = SDR((size, basal_ganglia.segments_per_cell)),
            permanence_thresh   = args.permanence_thresh,)

        self.active  = SDR((size,))

    def compute(self):
        """
        Computes each neurons excitement.  This saves all the intermediate
        state which is needed for learning later.
        """
        # Determine which segments activated and count them.
        self.segment_excitement = self.synapses.compute()
        self.active_segments    = self.segment_excitement >= self.args.predictive_threshold
        self.excitement         = np.sum(self.active_segments, axis=1)
        # Break ties randomly.
        self.excitement         = self.excitement + np.random.uniform(0, .5, size=self.size)
        return self.excitement

    # TODO: Move this method back to the BG class, it doesn't belong here.  Consider merging this method with the TD_Error method.
    def imbalance(self, td_error):
        """
        The imbalance is the number of additional D1 neurons and fewer D2
        neurons which should have activated to predict the correct value.
        This method discards the sign of TD Error and returns the absolute 
        magnitude of the imbalance.
        """
        k = self.args.num_neurons * self.args.sparsity
        return int(round(abs(td_error) * k / 2))   # Is okay? Guess and check.

    def strengthen(self, td_error):
        """
        Strengthen this population.  This chooses imbalance many neurons to
        learn about the current input pattern, only the most excited
        inactive neurons are chosen to learn.
        """
        imbalance     = self.imbalance(td_error)
        if imbalance == 0:
            return
        actual_active = len(self.active)
        target_active = actual_active + imbalance
        burst         = np.argpartition(-self.excitement, (actual_active, target_active))
        burst         = burst[actual_active : target_active]
        # All segments on bursting neurons which meet or excede the learning
        # threshold will learn.
        learning_segments_dense    = self.segment_excitement[burst] >= self.args.learning_threshold
        learn_cell_num, learn_segs = np.nonzero(learning_segments_dense)
        # The following line undoes the effect of selecting bursting neurons
        # before the nonzero operation.
        learn_cells = burst[learn_cell_num]
        self.synapses.learn_outputs(output_sdr     = (learn_cells, learn_segs),
                                    permanence_inc = self.args.permanence_inc,
                                    permanence_dec = self.args.permanence_dec,)

        # Add synapses to bursting neuron segments.  Specifically, subthreshold
        # learning segments should receive more synapses.
        subthreshold_learning_segs = np.logical_and(
                                            learning_segments_dense,
                                            np.logical_not( self.active_segments[burst] ))
        add_syn_cell_num, add_syn_seg = np.nonzero(subthreshold_learning_segs)
        add_syn_cell = burst[add_syn_cell_num] # Fix nonzero after removing unwanted data.
        self.synapses.add_synapses( output_sdr          = (add_syn_cell, add_syn_seg),
                                    synapses_per_output = self.args.add_synapses,
                                    maximum_synapses    = self.args.synapses_per_segment)

        # Give bursting neurons with no learning segments more synapses on new
        # segments.  Pick segments with few existing potential synapses on them,
        # which should make sure the segments are used to their full potential,
        # and not clumping too many synapses on too few segments.
        # TODO: This could count the number of *connected* inputs?
        need_more_segs = np.sum(subthreshold_learning_segs, axis=1) < 2 # Hardcoded Number!
        need_more_segs = burst[need_more_segs]
        if len(need_more_segs):
            sources    = self.synapses.postsynaptic_sources.reshape(self.synapses.outputs.dimensions)
            # Sources shape = (cells, segments)
            syn_counts = np.array([[seg.shape[0] for seg in seg_slice]
                                        for seg_slice in sources[need_more_segs]])
            syn_counts = syn_counts + np.random.uniform(0, .5, size=syn_counts.shape)   # break ties randomly
            new_seg    = np.argmin(syn_counts, axis=1)
            self.synapses.add_synapses( output_sdr          = (need_more_segs, new_seg),
                                        synapses_per_output = self.args.initial_segment_size,
                                        maximum_synapses    = self.args.synapses_per_segment)

    def weaken(self, td_error):
        """
        Weaken this population.  This supresses the least excited but
        still active neurons, penalizing all learning segments on those
        neurons so they will ignore this input pattern in the future.
        """
        imbalance          = self.imbalance(abs(td_error))
        if imbalance == 0:
            return
        # Find which neurons to supress.
        actual_active      = len(self.active)
        target_active      = max(0, actual_active - imbalance)
        active_excitement  = self.excitement[self.active.flat_index]
        partitioned_active = np.argpartition(-active_excitement, target_active)
        supress_active_num = partitioned_active[target_active:]
        allow_active_num   = partitioned_active[:target_active]
        # The following lines undo the effect of selecting the active neurons
        # before the argpartition.
        supress            = self.active.flat_index[supress_active_num]
        allow              = self.active.flat_index[allow_active_num]
        # self.active.flat_index = allow

        # Find which segments on the supressed neurons to weaken.
        learning_segments = self.segment_excitement[supress] >= self.args.learning_threshold
        learn_cell_num, lean_segs = np.nonzero(learning_segments)
        # The following line undoes the effect of selecting the supressed
        # neurons before the nonzero operation.
        learn_cells = supress[learn_cell_num]
        self.synapses.learn_outputs(output_sdr     = (learn_cells, lean_segs),
                                    permanence_inc = -self.args.permanence_dec,
                                    permanence_dec = 0,)

    def copy(self):
        """
        Make a shallow copy of this population so that it can learn, next cycle
        when the TD Error is known.
        """
        cpy          = copy.copy(self)
        cpy.synapses = self.synapses.copy()
        cpy.active   = SDR(self.active)
        return cpy

class BasalGangliaParameters(Parameters):
    parameters = [
        'permanence_dec',
        'permanence_inc',
        'permanence_thresh',
        'mispredict_dec',
        'predictive_threshold',
        'learning_threshold',
        'd1d2_ratio',
        'num_neurons',
        'apical_segments',
        'segments_per_cell',
        'sparsity',
        'synapses_per_segment',
        'td_lambda',
        'future_discount',
        'apical_segment_excite',
        'apical_segment_inhibit',
        'apical_voluntary',
        'apical_inhibit',
        'add_synapses',             # How many synapses to add to subthreshold learning segments.
        'initial_segment_size',     # How many synases to start new segments with.
    ]

class BasalGanglia:
    comment = """
        The Basal Ganglia (B.G.) performs reinforcement learning using the TD-Lambda
        algorithm.  This section attempts to explain the algorithm.  When B.G.
        neurons activate they learn, locking onto input patterns, and they compete
        with other B.G. neurons in much the same way as spatial pooler mini-columns
        do.  Within the B.G. there are two subtypes of neurons, called D1 and D2.
        The purpose of D1 neurons is to learn about positive rewards and D2 neurons
        learn about negative rewards (penalties).  Put together, D1 and D2 neurons
        form an estimate the expected value (summation) of rewards in the near
        future as well as a stabile representation of the current reward state.  It
        is called an estimate but it is also a prediction of the future.  How far
        into the future neurons concern themselves with is determined by the
        parameter 'future_discount'.  The expected value is a function of the number
        of active D1 cells and the number of active D2 cells.

        Every cycle the B.G. makes a new estimate/prediction of the expected value,
        adds to it any rewards which were received in the time between cycles, and
        compares it to the old estimate.  The difference between the old and new
        estimates is called the TD-Error.  A positive TD-Error indicates an
        unexpected reward while a negative TD-Error indicates an unexpected penalty.
        Notice that a reward is as good as the expectation of a reward, and that the
        new estimate is provably more accurate than the old one because the new
        estimate includes information about actual rewards received.

        D1 neurons only learn when the TD-Error is positive and D2 neurons only
        learn when the TD-Error is negative, and in both cases the learning rate is
        proportional to the absolute magnitude of the TD-Error.  In this way D1 and
        D2 neurons learn to recognise different events given the same inputs.
        Because B.G. neurons are not representing the current state of anything, but
        rather a prediction of the future, they do not learn when they activate.
        Instead they learn for a period of time immediately after each activation.
        During this learning period B.G. neurons are essentially checking their
        predictions.  When the TD-Error is non-zero, indicating an unexpected
        reward, some recently active B.G. neurons will learn to recognise the
        current input on the assumption that they probably should have seen the
        reward comming, and the next time this unexpected situtation occurs some of
        the neurons which learned about it should activate and improve the expected
        value estimate, ultimately leading to a reduced TD-Error.

        B.G. neurons use eligability traces to keep track of how recently they
        activated and if they can learn.  Active neurons set their eligability trace
        to '1' and all traces exponentially decay towards zero.  Each neurons
        learning rate is multiplied by its eligability trace, which implements the
        window of time following each activation when a neuron can learn corrections
        based on the TD-Error signal.  The parameter 'td_lambda' controls how fast
        traces decay.  TD-Lambda == 0 causes neurons to only learn in the next cycle
        and TD-Lambda == 1 causes neurons learn forever.  Conceptually, a neurons
        trace indicates how relevent it is to the current situtation.

        The TD-Error determines how many neurons in each population learn.  If a
        population should have had more activations, then the most excited sub-
        threshold neurons burst.  If a population should have had fewer activations,
        then the least excited active neurons are penalized.

        When either population has too few recent activations to learn about the TD-
        Error, then additional neurons are choosen to represent it.  These neurons
        burst and immediately learn.

        Bursting.  If there aren't enough neurons active for the TD-Error to
        influence then more neurons are chosen to activate and immediately represent
        the current input.  For example if there are no D2 neurons active (cause its
        been a good day) but then anything bad happens, then there are no eligable
        D2 neurons to learn...
            - Which neurons? the more excited ones are prefered.
            - When? Under what conditions?
            - How many?

        Add synapses ... Maybe to only bursting neurons
            - when & which neurons & which segments

        Mispredict
        Penalty ... active segments on inactive neurons are weakened?

        Representational Power ... talk about what this does to ensure that all of
        the available neurons are used in distinctive ways.
            - Boosing? track activation freq?
            - This is a measurable output of the B.G., how to test it?

        Finally talk about the output synapses ... IS THE FOLLOWING CORRECT?
        Adapt apical dendrites.  The Basal Ganglia outputs bias the agent's actions
        and the BG uses them to maximize its reward.  The given outputs are the L5
        activity of the previous cycle, which indirectly caused/affected the current
        inputs, reward, and TD-Error.  Adapt last cycle's active outputs to
        associate the recent BG activity (which is stored in the elegability traces)
        with the TD-Error that the output activations yielded. Do this step before
        incorperating the current BG activity into the traces. This way the output
        is learning to recognise the BG state at the time when it took action, not
        after???
        """
    def __init__(self, parameters, input_sdr, output_sdr):
        """
        Argument input_sdr is an instance of SDR, state of world to determine
                 expected value from.
        Argument output_sdr is an instance of SDR, L5 layer activations, these
                 are biased by the basal ganglia.  Both input_sdr and output_sdr
                 should be computed in the same cycle.
        """
        assert(isinstance(parameters, BasalGangliaParameters))
        assert(isinstance(input_sdr, SDR))
        assert(isinstance(output_sdr, SDR))
        self.args = args        = parameters
        self.inputs             = input_sdr
        self.outputs            = output_sdr
        self.apical_control     = (SDR(output_sdr.dimensions), SDR(output_sdr.dimensions))
        self.num_neurons        = int(round(args.num_neurons))
        self.segments_per_cell  = int(round(args.segments_per_cell))
        self.apical_segments    = int(round(args.apical_segments))
        num_d1                  = int(round(self.num_neurons * args.d1d2_ratio))
        num_d2                  = self.num_neurons - num_d1
        self.d1                 = StriatumPopulation(self, num_d1)
        self.d2                 = StriatumPopulation(self, num_d2)
        self.reset()

    def reset(self):
        # self.d1_traces = np.zeros(self.num_d1, dtype=PERMANENCE)
        # self.d2_traces = np.zeros(self.num_d2, dtype=PERMANENCE)
        self.expected_value = None
        self.td_error       = 0     # Is used for diagnostics.
        self.apical_control[0].zero()
        self.apical_control[1].zero()
        self.d1_inc = 0
        self.d1_dec = 0
        self.d2_inc = 0
        self.d2_dec = 0

    def expected_value_func(self, d1, d2):
        """
        Calculate the expected value. Sungur keeps track of each neurons value,
        which is updated after learning.  I want to take the ratio of active
        d1/d2 cells because its simpler and I think that assuming that some
        cells have values other than their binary activations is not going to
        work. Striatum cells fire under numerous circumstances, theres no way
        one value is going to work, and averaging many won't help. arctan2(d1,
        d2) will bring the expected value to the range [0, pi/2] The expected
        value needs to gel with the rewards, which are not bounded to any
        particular range or units.  I could rescale the rewards based on what
        keeps the expected value in a happy range?  I could use a genetic
        parameter to scale the reward to a range which works.

        fraction = (|D1| - |D2|) / (|D1| + |D2|)        # Range [-1, 1]
        theta    = fraction * pi/2                      # Range [-pi/2, pi/2]
        value    = tan(theta) * RewardSensitivity       # Range [-inf, inf]
        """
        d1 = len(d1)
        d2 = len(d2)
        new_expected_value = (d1 - d2) / (d1 + d2)
        return new_expected_value

    # TODO: Consider adding boosting, like SP does.  This could help ensure that
    # all BG neurons are used.
    def compute(self, reward):
        """
        Argument reward ... set to None to disable learning.
        """
        # Activate the neurons with the most active segments.
        d1_excitement = self.d1.compute()
        d2_excitement = self.d2.compute()
        segments      = np.concatenate([d1_excitement, d2_excitement])
        k             = max(1, int(round(self.args.sparsity * self.num_neurons)))
        active        = np.argpartition(-segments, k)[:k]
        self.d1.active.flat_index = active[active <  self.d1.size]
        self.d2.active.flat_index = active[active >= self.d1.size] - self.d1.size

        new_expected_value = self.expected_value_func(self.d1.active, self.d2.active)

        # Learn.
        if reward is not None and self.expected_value is not None:
            td_error = (reward + new_expected_value * self.args.future_discount) - self.expected_value
            self.td_error = td_error        # Is used for diagnostics.

            """
            EXPERIMENT:  TD-Error = sign(TD-Error) * log(1 + abs(TD-Error))
            Clean up TD-Error.  Currently TD-Error is a function of the expected
            values and the reward, all of which can have arbitrary ranges.  TD-
            Error is then used to modify all of the synapses by an undetermined
            amount.  I think the safest thing to do is take the logarithm of TD-
            Error because it almost always in a good range. Conceptually, the
            logarithm converts TD-Error into the scale that would be needed to
            represent it, in base e of course.  After this transform, the TD-
            Error is sensitive to percent changes in the reward/expected value
            instead of additive/subtractive changes.
            """
            # The sign and magitude of the TD-Error determine which population
            # (D1/D2) learns and how fast.  Only active neurons learn (or
            # recently active, as recorded in the elegability traces), and
            # within a learning neuron all segments which meet the
            # learning_threshold are adapted to recognise the current input.
            if td_error > 0:
                self.prev_d1.strengthen(td_error)
                self.prev_d2.weaken(td_error)
            elif td_error < 0:
                self.prev_d2.strengthen(-td_error)
                self.prev_d1.weaken(-td_error)

            # Apical synapses learn.  Apical synapses are managed by the TM
            # class, save as attributes the necessary permanence update
            # information for the TM class to use.
            inc = self.args.permanence_inc
            dec = self.args.permanence_dec
            if td_error > 0:
                self.d1_inc = inc * abs(td_error)
                self.d1_dec = dec * abs(td_error)
                self.d2_inc = - dec * abs(td_error)
                self.d2_dec = 0
            else:
                self.d1_inc = - dec * abs(td_error)
                self.d1_dec = 0
                self.d2_inc = inc * abs(td_error)
                self.d2_dec = dec * abs(td_error)

            # Apical synapse timing:
            # Cycle 1: Basal Ganglia neurons activate.
            # Cycle 2: Apical dendrites in the cortex see the new BG activations.
            #          Basal Ganglia neurons see the cortical activations which
            #          contain info about the current state of the world as well
            #          as the current actions.
            #          Basal Ganglia makes a new value estimate.
            #          BG uses TD error to adjust the apical synapses such that
            #          last cycle's activity causes this cycles (positive TD Error case)

        # Save (shallow) copies of the D1 and D2 population objects and other
        # things which are needed for learning.
        self.prev_d1                  = self.d1.copy()
        self.prev_d2                  = self.d2.copy()
        self.expected_value           = new_expected_value

    def statistics(self):
        # TODO: The TD-Error is a measure of how well the BG is representing the
        # expected value, and I should monitor it throughout the training to
        # show that it decreases with learning.
        stats = 'Basal Ganglia Statistics\n'
        stats += 'Avg TD Error %g\n'%self.mean_td_error
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
