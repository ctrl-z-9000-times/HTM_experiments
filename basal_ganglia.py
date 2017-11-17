# Written by David McDougall, 2017

import numpy as np
from sdr import SDR, Dendrite, WeightedSynapseManager
from genetics import Parameters
import copy

"""
System timing:
1) Encode sensory input.
2) Thalamus uses prior GP output to modulate cortical input.
3) Cortex computes.
4) Striatum computes.
5) GP computes expected value.
6) GP learns from TD-Error.
7) Striatum learns from (reward + current-EV * future-discount)
8) Cortex learns.
9) Motor controls compute, apply, and learn.

The cortex might have a stabilization mechanism which can see a single cycle into the past.
The striatum operates instantaneously (no time delays).
The GP has a single cycle delay before learning so that it can asses the TD-Error of its estimate.
"""

# TODO: Consider adding boosting, like SP does.  This could help ensure that
# all BG neurons are used.

class StriatumParameters(Parameters):
    parameters = [
        'add_synapses',
        'initial_segment_size',
        'learning_threshold',
        'mispredict_dec',
        'permanence_dec',
        'permanence_inc',
        'permanence_thresh',
        'predictive_threshold',
        'segments_per_cell',
        'sparsity',
        'synapses_per_segment',
    ]

class StriatumPopulation:
    """
    This class models the D1 and the D2 populations of neurons in the striatum of
    the basal ganglia.
    """
    def __init__(self, parameters, cortex_sdr, size,
        radii = None,):
        assert(radii is None) # TODO: Striatum should allow topology.
        self.args = args = parameters
        self.size        = size
        self.active      = SDR((size,), activation_frequency_alpha=0.005)
        self.synapses    = Dendrite(
            input_sdr            = cortex_sdr,
            active_sdr           = self.active,
            segments_per_cell    = args.segments_per_cell,
            synapses_per_segment = args.synapses_per_segment,
            predictive_threshold = args.predictive_threshold,
            learning_threshold   = args.learning_threshold,
            permanence_thresh    = args.permanence_thresh,
            permanence_inc       = args.permanence_inc,
            permanence_dec       = args.permanence_dec,
            mispredict_dec       = args.mispredict_dec,
            add_synapses         = args.add_synapses,
            initial_segment_size = args.initial_segment_size,)

    def compute(self, cortex_sdr=None):
        self.excitement         = self.synapses.compute(input_sdr=cortex_sdr)
        self.excitement         = self.excitement + np.random.uniform(0, .5, size=self.size)
        k                       = max(1, int(round(self.args.sparsity * self.size)))
        self.active.flat_index  = np.argpartition(-self.excitement, k)[:k]
        return self.active

    def learn(self):
        """Caller must gate this method on the sign of the world current value."""
        self.synapses.learn()

    def copy(self):
        cpy          = copy.copy(self)
        cpy.synapses = self.synapses.copy()
        cpy.active   = cpy.synapses.active_sdr
        return cpy

    def statistics(self):
        s = self.synapses.statistics()
        s += 'Active ' + self.active.statistics()
        return s


class GlobusPallidusParameters(Parameters):
    parameters = [
        'permanence_dec',
        'permanence_inc',
        'predictive_threshold',
        'learning_threshold',
        'num_neurons',
        'segments_per_cell',
        'synapses_per_segment',
        'sparsity',
        'add_synapses',             # How many synapses to add to subthreshold learning segments.
        'initial_segment_size',     # How many synases to start new segments with.
    ]

class GlobusPallidus:
    """
    GP neurons have numerous segments containing variable strength synapses
    from striatal neurons.  GP Neurons active when they win out the competition
    for segment detections.  They then inhibt each other to control sparsity.
    """
    def __init__(self, parameters, striatum_sdr):
        """
        """
        assert(isinstance(parameters, GlobusPallidusParameters))
        self.args = args        = parameters
        self.striatum           = striatum_sdr
        self.num_neurons        = int(round(args.num_neurons))
        self.active             = SDR((self.num_neurons,))
        self.synapses           = WeightedSynapseManager(
            input_sdr         = self.striatum,
            output_sdr        = SDR((self.num_neurons, int(round(args.segments_per_cell)),)),
            permanence_thresh = .5, # Unused parameter.
            permanence_inc    = args.permanence_inc,
            permanence_dec    = args.permanence_dec,)

    def compute(self, input_sdr=None):
        self.segment_excitement = self.synapses.compute(input_sdr=input_sdr)
        self.active_segments    = self.segment_excitement >= self.args.predictive_threshold
        self.excitement         = np.sum(self.active_segments, axis=1)
        self.excitement         = self.excitement + np.random.uniform(0, .5, size=self.num_neurons) # Break ties randomly.
        return self.excitement

    def imbalance(self, td_error):
        """
        The imbalance is the number of additional GPi neurons and fewer GPe
        neurons which should have activated to predict the correct value.
        """
        k = (self.args.num_neurons) * self.args.sparsity
        return int(round(td_error * k))   # Is okay?

    def strengthen(self, td_error):
        """
        Strengthen this population by activating additional neurons and causing
        them to learn the current input pattern.
        """
        imbalance     = abs(self.imbalance(td_error))
        if imbalance == 0:
            return
        td_error      = abs(td_error)   # Maybe log(1 + abs(td)) ?
        actual_active = len(self.active)
        target_active = actual_active + imbalance
        partitioned   = np.argpartition(-self.excitement, (actual_active, target_active))
        burst         = partitioned[actual_active : target_active]
        self.active.flat_index = partitioned[:target_active]
        # All segments on bursting neurons which meet or excede the learning
        # threshold will learn.
        learning_segments_dense    = self.segment_excitement[burst] >= self.args.learning_threshold
        learn_cell_num, learn_segs = np.nonzero(learning_segments_dense)
        # The following line undoes the effect of selecting bursting neurons
        # before the nonzero operation.
        learn_cells = burst[learn_cell_num]
        self.synapses.learn_outputs(output_sdr     = (learn_cells, learn_segs),
                                    permanence_inc = td_error * self.args.permanence_inc,
                                    permanence_dec = td_error * self.args.permanence_dec,)

        # Add synapses to bursting neuron segments.  Specifically, subthreshold
        # learning segments should receive more synapses.
        subthreshold_learning_segs = np.logical_and(
                                            learning_segments_dense,
                                            np.logical_not( self.active_segments[burst] ))
        add_syn_cell_num, add_syn_seg = np.nonzero(subthreshold_learning_segs)
        add_syn_cell = burst[add_syn_cell_num] # Fix nonzero after removing unwanted data.
        self.synapses.add_synapses( output_sdr          = (add_syn_cell, add_syn_seg),
                                    synapses_per_output = self.args.add_synapses,
                                    maximum_synapses    = self.args.synapses_per_segment,
                                    init_value          = td_error * self.args.permanence_inc)

        # Give bursting neurons with no learning segments more synapses on new
        # segments.  Pick segments with fewest existing synapses on them, which
        # should make sure the segments are used to their full potential, and
        # not clumping too many synapses on too few segments.
        need_more_segs = np.logical_not(np.any(subthreshold_learning_segs, axis=1))
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
                                        maximum_synapses    = self.args.synapses_per_segment,
                                        init_value          = td_error * self.args.permanence_inc)

    def weaken(self, td_error):
        """
        Weaken this population by supressing some neurons and causing them to
        unlearn the current input pattern.
        """
        imbalance          = abs(self.imbalance(td_error))
        if imbalance == 0:
            return
        td_error           = abs(td_error)   # Maybe log(1 + abs(td)) ?
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
        self.active.flat_index = allow

        # Find which segments on the supressed neurons to weaken.
        learning_segments = self.segment_excitement[supress] >= self.args.learning_threshold
        learn_cell_num, lean_segs = np.nonzero(learning_segments)
        learn_cells = supress[learn_cell_num]   # Fix nonzero after select.
        self.synapses.learn_outputs(output_sdr     = (learn_cells, lean_segs),
                                    permanence_inc = -1 * td_error * self.args.permanence_dec,
                                    permanence_dec = 0,)

    def copy(self):
        cpy          = copy.copy(self)
        cpy.active   = SDR(self.active)
        cpy.synapses = self.synapses.copy()
        return cpy


class BasalGangliaParameters(Parameters):
    parameters = [
        'd1d2_ratio',
        'gp',
        'num_msn',
        'striatum',
        # 'future_discount',
    ]

class BasalGanglia:
    """
    This class combines a D1, D2, GPe, and GPi into a single functional unit.
    """
    def __init__(self, parameters, cortex_sdr, future_discount):
        self.args = args = parameters
        args.future_discount = future_discount
        num_msn   = int(round(args.num_msn))
        num_d1    = int(round(num_msn * args.d1d2_ratio))
        self.d1   = StriatumPopulation(args.striatum, cortex_sdr, num_d1)
        self.d2   = StriatumPopulation(args.striatum, cortex_sdr, num_msn - num_d1)
        self.gpi  = GlobusPallidus(args.gp, self.d1.active)
        self.gpe  = GlobusPallidus(args.gp, self.d2.active)
        self.reset()

    def reset(self):
        self.expected_value = None
        self.td_error       = None

    def compute(self, cortex_sdr, reward):
        """
        Argument reward ... set to None to disable learning.
        """
        # Compute Striatum activity.
        self.d1.compute(cortex_sdr)
        self.d2.compute(cortex_sdr)
        # Compute Globius Pallidus activity.
        gpi_excitement = self.gpi.compute()
        gpe_excitement = self.gpe.compute()
        excitement     = np.concatenate([gpi_excitement, gpe_excitement])
        k              = max(1, int(round(self.args.gp.sparsity * 2 * self.args.gp.num_neurons)))
        active         = np.argpartition(-excitement, k)[:k]
        self.gpi.active.flat_index = active[active <  self.gpi.active.size]
        self.gpe.active.flat_index = active[active >= self.gpi.active.size] - self.gpi.active.size

        new_expected_value = self.expected_value_func(self.gpi.active, self.gpe.active)

        # Learn about the previous cycle from the reward.
        if reward is not None and self.expected_value is not None:
            updated_estimated_EV = reward + new_expected_value * self.args.future_discount
            self.td_error        = updated_estimated_EV - self.expected_value

            if self.td_error > 0:
                self.prev_gpi.strengthen(self.td_error)
                self.prev_gpe.weaken(self.td_error)
            elif self.td_error < 0:
                self.prev_gpe.strengthen(-self.td_error)
                self.prev_gpi.weaken(-self.td_error)

            if updated_estimated_EV > 0:
                self.prev_d1.learn()
            elif updated_estimated_EV < 0:
                self.prev_d2.learn()
        # Save these things for learning next cycle.
        self.expected_value = new_expected_value
        self.prev_gpi       = self.gpi.copy()
        self.prev_gpe       = self.gpe.copy()
        self.prev_d1        = self.d1.copy()
        self.prev_d2        = self.d2.copy()

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

    def statistics(self):
        stats = 'Basal Ganglia Statistics\n'
        stats += 'TD Error %g\n'%self.td_error
        stats += 'D1 ' + self.d1.statistics()   + '\n'
        stats += 'D2 ' + self.d2.statistics()   + '\n'
        stats += 'GPi ' + self.gpi.synapses.statistics() + '\n'
        stats += 'GPe ' + self.gpe.synapses.statistics() + '\n'
        return stats
