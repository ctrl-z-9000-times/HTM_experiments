# Written by David McDougall, 2017
# cython: language_level=3
# cython: profile=True
"""
BUILD COMMAND
./setup.py build_ext --inplace
"""
cimport numpy as np
cimport cython

import numpy as np
from genetics import Parameters
from sdr import SDR, SynapseManager, Dendrite
import math

class UnifiedParameters(Parameters):
    """Example..."""
    parameters = [
        'AP',
        'boosting_alpha',
        'target_sparsity',
        'cells',
        'distal_add_synapses',             # How many new synapses to add to subthreshold learning segments.
        'distal_init_dist',                # Initial distribution of permanence values.
        'distal_initial_segment_size',     # How many synases to start new segments with.
        'distal_learning_threshold',       # Segment excitement threshold for learning.
        'distal_mispredict_dec',
        'distal_permanence_dec',
        'distal_permanence_inc',
        'distal_permanence_thresh',
        'distal_predicted_boost',          # Predicted cells activate this many times.
        'distal_predictive_threshold',     # Segment excitement threshold for predictions.
        'distal_segments_per_cell',
        'distal_synapses_per_segment',
        'min_stability',                    # float in range [0, 1) or None
        'predicted_proximal_boost',
        'proximal_add_synapses',
        'proximal_active_thresh',
        'proximal_dec',
        'proximal_inc',
        'proximal_init_dist',
        'proximal_potential_pool',          # TODO: Rename to 'proximal_synapses_per_segment'
        'proximal_segments',
        'proximal_thresh',
    ]
    def min_sparsity(self):
        return math.ceil(self.AP / self.distal_predicted_boost) / self.cells
    def max_sparsity(self):
        return self.AP / self.cells

class Unified:
    """
    """
    def __init__(self, parameters, input_sdr, context_sdr, macro_columns, radii=tuple()):
        assert(isinstance(parameters, UnifiedParameters))
        assert(isinstance(input_sdr, SDR))
        assert(isinstance(context_sdr, SDR))
        self.args          = args = parameters
        self.input_sdr     = input_sdr
        self.context_sdr   = context_sdr
        self.macro_columns = tuple(int(round(dim)) for dim in macro_columns)
        self.active        = SDR(self.macro_columns + (args.cells,),
                                activation_frequency_alpha = 1/1000,)
        self.age           = 0

        # Now build the synapse manager and dendrite instances
        px_seg_shape  = self.active.dimensions + (int(round(args.proximal_segments)),)
        self.proximal = SynapseManager( self.input_sdr,
                                        SDR(px_seg_shape),
                                        radii             = radii,
                                        init_dist         = args.proximal_init_dist,
                                        permanence_inc    = args.proximal_inc,
                                        permanence_dec    = args.proximal_dec,
                                        permanence_thresh = args.proximal_thresh,)

        self.proximal.normally_distributed_connections(args.proximal_potential_pool, radii)

        if args.boosting_alpha is not None:
            # Make a dedicated SDR to track segment activation frequencies for
            # boosting.
            self.boosting = SDR(self.proximal.outputs,
                                activation_frequency_alpha = args.boosting_alpha)
            # Initialize to the target activation frequency/sparsity.
            self.boosting.activation_frequency.fill(args.target_sparsity)

        if args.min_stability is not None:
            self.prev_active = SDR(self.active)

        self.basal = Dendrite(
            input_sdr            = context_sdr,
            active_sdr           = SDR(self.active),
            segments_per_cell    = args.distal_segments_per_cell,
            synapses_per_segment = args.distal_synapses_per_segment,
            initial_segment_size = args.distal_initial_segment_size,
            add_synapses         = args.distal_add_synapses,
            learning_threshold   = args.distal_learning_threshold,
            predictive_threshold = args.distal_predictive_threshold,
            permanence_inc       = args.distal_permanence_inc,
            permanence_dec       = args.distal_permanence_dec,
            permanence_thresh    = args.distal_permanence_thresh,
            mispredict_dec       = args.distal_mispredict_dec,
            init_dist            = args.distal_init_dist,)

        self.reset()

    def reset(self):
        if self.args.min_stability is not None:
            self.prev_active.zero()
        self.active.zero()
        self.reset_state = True

    def compute(self):
        args = self.args
        cdef:
            np.ndarray[np.double_t, ndim=2] neuron_excitment
            int cells = args.cells
            int AP    = args.AP
            int active_thresh   = args.proximal_active_thresh
            int predicted_boost = args.distal_predicted_boost
            int num_mcols
            int mcol, index, flat_index
            int inhibition, v
            int pred
            int anomaly_count = 0
            np.ndarray[np.int_t]   predictions_flat
            np.ndarray[np.int_t]   stable_ranking
            np.ndarray[np.int_t]   ranking
            np.ndarray[np.int_t]   activations
            np.ndarray[np.uint8_t] nz_values
            int total_activations, mcol_activations
            int target_stable
            int stable_cells, stability, cell_num
            np.ndarray[np.uint8_t, ndim=2] prev_active_shaped = None
            np.ndarray[np.int_t]           prev_active_slice

        if args.min_stability is not None:
            target_stable = int(round(AP * args.min_stability))
            prev_active_shaped = self.prev_active.dense.reshape(-1, args.cells)
        else:
            target_stable = 0

        predictions      = self.basal.compute()
        predictions_flat = predictions.reshape(-1)

        excitment = self.proximal.compute()
        # Break ties randomly.
        excitment = excitment + np.random.uniform(0, 1, size=excitment.shape)
        # Logarithmic Boosting Function.
        if args.boosting_alpha is not None:
            boost = np.log2(self.boosting.activation_frequency) / np.log2(args.target_sparsity)
            excitment *= np.nan_to_num(boost)

        # TODO: Instead of saving this, save the argmax of segment excitement
        # which is what is used for learning.  And fold it into the next step...
        self.segment_excitement = excitment     # Needed for learning.

        # Reduce each neurons proximal segments to its most excited segment.
        excitment = np.max(excitment, axis=-1)

        # Estimate the time at which each proximal dendrite crosses its
        # activations threshold.
        thresh = args.proximal_active_thresh
        beta   = args.predicted_proximal_boost
        thresholds = thresh * (1 - beta * (predictions != 0))
        # TODO: Rename this variable. This is not the excitement.  This is the
        # estimated time when the the proximal segments activate.  Maybe copy-
        # paste some of the explaination from the lab report into here.
        excitment  = thresholds / excitment

        # Activate cells on the basis of how excited they are.
        neuron_excitment = excitment.reshape(-1, args.cells) # Flatten all but the last dimension

        # Iterate through the macro columns
        num_mcols   = neuron_excitment.shape[0]
        activations = np.empty(AP * num_mcols, dtype=np.int)
        nz_values   = np.empty(AP * num_mcols, dtype=np.uint8)
        total_activations = 0
        for mcol in range(num_mcols):
            inhibition = 0

            # This activates previously active cells in order to maintain
            # (args.min_stability) percent overlap between time steps.
            if target_stable > 0:
                prev_active_slice = np.nonzero(prev_active_shaped[mcol, :])[0]
                stable_ranking    = np.argsort(neuron_excitment[mcol][prev_active_slice])
                stable_cells      = 0
                stability         = 0
                while       (stability   < target_stable 
                        and stable_cells < stable_ranking.shape[0]
                        and inhibition   < AP):
                    cell_num   = stable_ranking[stable_cells]
                    index      = prev_active_slice[cell_num]
                    # Check for proximal segments which are below the activation threshold.
                    if neuron_excitment[mcol, index] > 1:
                        break
                    flat_index = index + mcol * cells
                    pred       = predictions_flat[flat_index]
                    if pred > 0:
                        v = predicted_boost
                    else:
                        v = 1
                        anomaly_count += 1
                    # Activate this neuron.
                    activations[total_activations] = flat_index
                    nz_values[total_activations]   = v
                    total_activations += 1
                    stable_cells      += 1
                    stability         += min(v, prev_active_shaped[mcol, index])
                    inhibition        += v
                    # Prevent this cell from activating durring the second loop.
                    neuron_excitment[mcol, index] = 9999

            # Activate cells regardless of the prior state of the world.
            ranking = np.argsort(neuron_excitment[mcol])
            mcol_activations = 0    # Poorly named...
            while inhibition < AP:
                index = ranking[mcol_activations]
                # Check for proximal segments which are below the activation threshold.
                if neuron_excitment[mcol, index] > 1:
                    break
                # Check if the neuron was in a predictive state.
                flat_index = index + mcol * cells
                pred       = predictions_flat[flat_index]
                if pred > 0:
                    v = predicted_boost
                else:
                    v = 1
                    anomaly_count += 1
                # Activate this neuron.
                activations[total_activations] = flat_index
                nz_values[total_activations]   = v
                total_activations += 1
                mcol_activations  += 1
                inhibition        += v
        self.active.flat_index = activations[:total_activations]
        self.active.nz_values  = nz_values[:total_activations]

        if args.min_stability is not None:
            self.stability = self.active.overlap(self.prev_active)
            # Force these lazy variables to construct before the assign method makes a copy.
            self.active.index; self.active.dense
            self.prev_active.assign(self.active)

        try:
            self.anomaly = anomaly_count / total_activations
        except ZeroDivisionError:
            self.anomaly = float('nan')

        return self.active

    def learn(self):
        """
        Make the spatial pooler learn about its current inputs and active columns.
        """
        self.age += 1
        # Select the most excited proximal segment on each active neuron to learn.
        segment_excitement = self.segment_excitement[self.active.index]
        seg_idx = np.argmax(segment_excitement, axis=-1)
        self.proximal.outputs.index = self.active.index + (seg_idx,)
        self.proximal.outputs.nz_values = self.active.nz_values
        self.proximal.learn()
        self.proximal.add_synapses(self.args.proximal_add_synapses)
        # Update the exponential moving average of each segments activation frequency.
        if self.args.boosting_alpha is not None:
            self.boosting.assign(self.proximal.outputs)

        # Learn about the previous to current timestep transition.
        if self.reset_state:
            # Learning on the first timestep after a reset is not useful, and
            # the data is uninitialized.
            self.reset_state = False
        else:
            self.basal.learn(active_sdr=self.active)

    # TODO: Add args.min_sparsity() and args.max_sparsity()
    def statistics(self):
        stats = ''
        stats += 'Activation statistics ' + self.active.statistics()
        stats += 'Proximal ' + self.proximal.statistics()
        if self.args.boosting_alpha is not None:
            stats      += 'Boosting ' + self.boosting.statistics()
            af         = self.boosting.activation_frequency
            target     = self.args.target_sparsity
            boost_min  = np.log2(np.min(af))  / np.log2(target)
            boost_mean = np.log2(np.mean(af)) / np.log2(target)
            boost_max  = np.log2(np.max(af))  / np.log2(target)
            stats += '\tLogarithmic Boosting Multiplier min/mean/max  {:-.04g}% / {:-.04g}% / {:-.04g}%\n'.format(
                    boost_min   * 100,
                    boost_mean  * 100,
                    boost_max   * 100,)
        stats += '\n'

        stats += 'Predictive Segments ' + self.basal.statistics()

        return stats
