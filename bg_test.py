#!/usr/bin/python3
"""
Written by David McDougall 2017

The purpose of this experiment is to test the basal ganglia system with out
needing actions or motors of any kind.

Ideas for interesting datasets:
1) Adjust reward magnitudes (but not the signs), this isolates learning to the GP.
    TODO: Striatum works.  Plot D1 & D2 classification accuracy as a function of the
    sequences value.  This plot should clearly show what each population knows and
    does not know.  Then randomly change the magnitude (but not the sign) of all
    rewards and retrain the network.  Show that the striatum's knowledge is
    unchanged, and that the GP has made the needed adjustments.
2) Adversarial situations which force the striatum or GP to do weird things.
    Make the GP compare many small rewards with a large punishment (and vise versa)
3) Generalization tests.  Build a simple language and generate sentences and see
    if the TM can learn the basic pattern well enough for the striatum to
    represent it and for the the GP to value it.
4) Capacity tests which take an existing test and scale it up.  Plot dataset
    size versus performance.  Discuss how TD-Lambda == 0 effects learning speed.


"""

import numpy as np
import htm
import itertools
import random
import genetics
from sdr import SDR
import basal_ganglia

class Dataset:
    """
    This is a timeseries dataset consisting of sequences leading up to rewards.
    The instantaneous inputs are randomly generated sets of neural activations.
    Only the final step of the sequence has a reward, all preceding steps have
    zero reward.  The rewards are uniform random numbers in the range [-1, 1].
    A random number of random inputs are inserted between each sequence, and
    which sequence is played is randomly chosen.
    """
    def __init__(self,
        num_sequences   = 50,
        sequence_length = range(4, 20),
        filler_length   = range(2, 4),):
        """
        Returns an iterator over the dataset which yields pairs of (input, reward).
        Nonzero reward are preceded by a sequence of contextual clues which should
        allow the agent to predict the reward.
        """
        self.filler_length = filler_length
        self.class_shape   = (num_sequences,)
        alphabet       = [chr(ord('A') + x) for x in range(26)]
        self.inputs    = [''.join(chrs) for chrs in itertools.product(alphabet, repeat=2)]
        self.sequences = []
        self.rewards   = []
        for i in range(num_sequences):
            self.sequences.append([])
            seq_len = random.randrange(min(sequence_length), max(sequence_length) + 1)
            for item in range(seq_len):
                self.sequences[-1].append(random.choice(self.inputs))
            self.rewards.append(random.uniform(-1, 1))
        self.state = ('filler', 1)

    def __next__(self):
        """
        This iterator is a FSM controlled by the attribute state.  State is a
        tuple and the first element of it is either 'sequence' or 'filler'.

        This sets attribute anomallous which indicates if the yielded values are
        predictable (False) or random (True).
        """
        if self.state[0] == 'sequence':
            seq  = self.state[1]
            step = self.state[2]
            if step == 0:
                self.anomallous = True
            else:
                self.anomallous = False
            if step + 1 >= len(self.sequences[seq]):
                remaining  = random.randrange(min(self.filler_length), max(self.filler_length) + 1)
                self.state = ('filler', remaining)
                return (self.sequences[seq][step], self.rewards[seq])
            else:
                self.state = ('sequence', seq, step + 1)
                return (self.sequences[seq][step], 0)

        elif self.state[0] == 'filler':
            remaining = self.state[1]
            if remaining - 1 <= 0:
                next_sequence = random.randrange(len(self.sequences))
                self.state = ('sequence', next_sequence, 0)
            else:
                self.state = ('filler', remaining - 1)
            self.anomallous = True
            return random.choice(self.inputs), 0

    def adjust_rewards(self):
        """Randomly adjust the magnitude (but not the sign) of all rewards."""
        for r, idx in enumerate(self.rewards):
            if r >= 0:
                self.rewards[idx] = random.uniform(0, 1)
            else:
                self.rewards[idx] = random.uniform(-1, 0)


class AdversarialDataset(Dataset):
    """
    This generates sequences with a common starting portion but different
    endings.  The purpose is to force the BG to compare many small rewards with
    a large punishment.
    """
    def __init__(self):
        super().__init__(num_sequences = 10,)
        self.sequences = []
        self.rewards = []
        common_start = [random.choice(self.inputs) for step in range(6)]
        for seq in range(10):
            ending_length = random.randrange(2, 5)
            ending = [random.choice(self.inputs) for step in range(ending_length)]
            self.sequences.append(common_start + ending)

        for seq in range(9):
            self.rewards.append(random.uniform(0, .5))      # Small Rewards
        for seq in range(1):
            self.rewards.append(random.uniform(-1, -.5))    # Large Punishment


class PatternDataset(Dataset):
    """
    Build a simple language and generate sentences and see if the TM can learn
    the basic pattern well enough for the striatum to represent it and for the
    the GP to value it.
    """
    def __init__(self):
        super().__init__(num_sequences = 100,)
        1/0 # Unimplemented.


class BG_Test(genetics.Individual):
    parameters = [
        'enc_bits',
        'enc_spar',
        'sp',
        'cols',
        'tm',
        'bg',
    ]
    fitness_names_and_weights = {
        'anom_rand':   +1,
        'anom_pred':   -1,
        'seq_sp':       0,
        'seq_tm':      +1,
        'seq_d1':       0,
        'seq_d2':       0,
        'td_error':    -2,
        'time':        -1,
        'memory':      -1,
    }

    def __init__(self):
        # Updated 11/17/2017 from test6
        self.bg = basal_ganglia.BasalGangliaParameters(
            d1d2_ratio = 0.5,
            gp = basal_ganglia.GlobusPallidusParameters(
                add_synapses         = 10,
                initial_segment_size = 28.668820802153853,
                learning_threshold   = 0.5,
                num_neurons          = 1477.8991746705415,
                permanence_dec       = 0.02,
                permanence_inc       = 0.14515840704318025,
                predictive_threshold = 1,
                segments_per_cell    = 100,
                sparsity             = 0.11932199421272612,
                synapses_per_segment = 88.46359570430134,),
            num_msn = 1400.7474298618658,
            striatum = basal_ganglia.StriatumParameters(
                add_synapses         = 3,
                initial_segment_size = 10,
                learning_threshold   = 5,
                mispredict_dec       = 0.001,
                permanence_dec       = 0.01,
                permanence_inc       = 0.03,
                permanence_thresh    = 0.25,
                predictive_threshold = 7,
                segments_per_cell    = 79.98172860414674,
                sparsity             = 0.04,
                synapses_per_segment = 47.3462661964499,),)
        self.cols = 2115.444367490216
        self.enc_bits = 3000
        self.enc_spar = 0.15
        self.sp = htm.SpatialPoolerParameters(
            boosting_alpha    = 0.0008051932203712806,
            permanence_dec    = 0.01057,
            permanence_inc    = 0.051968183603686754,
            permanence_thresh = 0.279,
            potential_pool    = 2500,
            sparsity          = 0.05,)
        self.tm = htm.TemporalMemoryParameters(
            add_synapses         = 10,
            cells_per_column     = 12,
            initial_segment_size = 20,
            learning_threshold   = 6,
            mispredict_dec       = 0.001051,
            permanence_dec       = 0.01,
            permanence_inc       = 0.03,
            permanence_thresh    = 0.2,
            predictive_threshold = 8,
            segments_per_cell    = 50,
            synapses_per_segment = 50,)

    debug  = False
    cycles = None   # Set by main function.
    def evaluate(self):
        datastream = Dataset()
        timer      = genetics.speed_fitness(20, 30)

        # SETUP AI.
        enc = htm.EnumEncoder(self.enc_bits, self.enc_spar, diag=False)
        sp  = htm.SpatialPooler(self.sp,
                                input_sdr  = enc.output_sdr,
                                column_sdr = SDR((self.cols,)),)
        sp_sdrc = htm.SDR_Classifier(htm.SDRC_Parameters(alpha = 0.001),
                                     sp.columns, datastream.class_shape, 'index')
        tm  = htm.TemporalMemory(self.tm, sp.columns)
        tm_sdrc = htm.SDR_Classifier(htm.SDRC_Parameters(alpha = 0.001),
                                     tm.active, datastream.class_shape, 'index')
        bg  = basal_ganglia.BasalGanglia(self.bg, tm.active, future_discount = .95)
        d1_sdrc = htm.SDR_Classifier(htm.SDRC_Parameters(alpha = 0.001),
                                     bg.d1.active, datastream.class_shape, 'index')
        d2_sdrc = htm.SDR_Classifier(htm.SDRC_Parameters(alpha = 0.001),
                                     bg.d2.active, datastream.class_shape, 'index')
        memory_score = genetics.memory_fitness(2e9, 3e9)

        # SETUP ACCUMULATORS.
        anom_rand = 0; anom_rand_total = 0
        anom_pred = 0; anom_pred_total = 0
        sp_seq_score = 0
        tm_seq_score = 0
        d1_seq_score = 0
        d2_seq_score = 0
        sequence_total = 0
        td_error = 0; td_error_total = 0    # RMS of TD-Error
        baseline = 0    # RMS(reward). If EV === 0 then this is also the RMS TD-Error.
        if self.debug:
            input_history = []
            reward_history = []
            anomalous_input_history = []
            ev_history = []
            td_error_history = []
            anomaly_history = []

        def plot_striatum_performance_vs_reward():
            # Measure classification accuracy of each sequence.
            d1_seq_scores = []
            d2_seq_scores = []
            for seq_idx, seq in enumerate(datastream.sequences):
                reward = datastream.rewards[seq_idx]
                d1_seq_scores.append(0)
                d2_seq_scores.append(0)
                seqence_classification_samples = 0
                for measurement in range(3):
                    # Add random inputs at the start of the seqence.
                    reset_steps = random.randrange(min(datastream.filler_length), max(datastream.filler_length) + 1)
                    reset_noise = [random.choice(datastream.inputs) for step in range(reset_steps)]
                    seq         = seq + reset_noise
                    for step, inp in enumerate(seq):
                        enc.encode(inp)
                        sp.compute()
                        tm.compute()
                        bg.compute(tm.active, reward=None)  # No learning.
                        # Filter out the random noise at the start of the sequence.
                        if step not in range(reset_steps):
                            d1_seq_cls = d1_sdrc.predict(bg.d1.active)
                            d2_seq_cls = d2_sdrc.predict(bg.d2.active)
                            d1_seq_scores[seq_idx] += d1_seq_cls[seq_idx] / np.sum(d1_seq_cls)
                            d2_seq_scores[seq_idx] += d2_seq_cls[seq_idx] / np.sum(d2_seq_cls)
                            seqence_classification_samples += 1
                d1_seq_scores[seq_idx] /= seqence_classification_samples
                d2_seq_scores[seq_idx] /= seqence_classification_samples
            # Plot the relationship between sequence value and which striatum
            # populations learned to recognise the sequence.
            from matplotlib import pyplot as plt
            plt.figure('Reward versus Striatum')
            plt.subplot(1, 2, 1)
            plt.title('D1')
            plt.plot(datastream.rewards, d1_seq_scores, 'ro')
            plt.xlabel('Sequence Reward')
            plt.ylabel('Classification Accuracy')
            plt.subplot(1, 2, 2)
            plt.title('D2')
            plt.plot(datastream.rewards, d2_seq_scores, 'bo')
            plt.xlabel('Sequence Reward')
            plt.ylabel('Classification Accuracy')

        # RUN ONLINE.
        for step in range(self.cycles):
            inp, reward = next(datastream)
            enc.encode(inp)
            sp.compute()
            tm.compute()
            bg.compute(tm.active, reward)
            sp.learn()
            tm.learn()

            # Measure performance.
            if bg.td_error is not None:
                baseline       += reward ** 2
                td_error       += bg.td_error ** 2
                td_error_total += 1

            if datastream.anomallous:
                anom_rand       += tm.anomaly
                anom_rand_total += 1
            else:
                anom_pred       += tm.anomaly
                anom_pred_total += 1

            # Train and test sequence classifiers for every part of the system.
            if datastream.state[0] == 'sequence':
                sp_seq_cls = sp_sdrc.predict(sp.columns)
                tm_seq_cls = tm_sdrc.predict(tm.active)
                d1_seq_cls = d1_sdrc.predict(bg.d1.active)
                d2_seq_cls = d2_sdrc.predict(bg.d2.active)
                sequence_total += 1
                seq_idx = datastream.state[1]
                # SDR Classifier outputs a PDF.  At creation, PDF may be beneath
                # the minumum representable floating point value.
                sp_seq_score += np.nan_to_num( sp_seq_cls[seq_idx] / np.sum(sp_seq_cls) )
                tm_seq_score += np.nan_to_num( tm_seq_cls[seq_idx] / np.sum(tm_seq_cls) )
                d1_seq_score += np.nan_to_num( d1_seq_cls[seq_idx] / np.sum(d1_seq_cls) )
                d2_seq_score += np.nan_to_num( d2_seq_cls[seq_idx] / np.sum(d2_seq_cls) )
                sp_sdrc.train(sp.columns,   (seq_idx,))
                tm_sdrc.train(tm.active,    (seq_idx,))
                d1_sdrc.train(bg.d1.active, (seq_idx,))
                d2_sdrc.train(bg.d2.active, (seq_idx,))

            if self.debug:
                # Record everything for debugging.
                input_history.append(inp)
                reward_history.append(reward)
                anomalous_input_history.append(datastream.anomallous)
                ev_history.append(bg.expected_value)
                td_error_history.append(bg.td_error if bg.td_error is not None else 0)
                anomaly_history.append(tm.anomaly)

        # REPORT.
        sp_seq_score = sp_seq_score / sequence_total
        tm_seq_score = tm_seq_score / sequence_total
        d1_seq_score = d1_seq_score / sequence_total
        d2_seq_score = d2_seq_score / sequence_total
        anom_pred    = anom_pred / anom_pred_total
        anom_rand    = anom_rand / anom_rand_total
        baseline     = (baseline / td_error_total) ** .5
        td_error     = (td_error / td_error_total) ** .5
        fitness      = {
            'anom_pred':    anom_pred,
            'anom_rand':    anom_rand,
            'seq_sp':       sp_seq_score,
            'seq_tm':       tm_seq_score,
            'seq_d1':       d1_seq_score,
            'seq_d2':       d2_seq_score,
            'td_error':     td_error / baseline,
            'time':         timer.done(),
            'memory':       memory_score,
        }
        if self.debug:
            print(sp.statistics())
            print(tm.statistics())
            print(bg.statistics())
            print("TD-Error Baseline", baseline, "Measured", td_error)
            print(fitness)

            plot_striatum_performance_vs_reward()

            # Plot the TD-Error and anomaly.
            from matplotlib import pyplot as plt
            steps = np.arange(len(input_history))
            plt.figure('Reinforcement Learning Graph')
            plt.title('Reward is Red, Expected Value is Green, TD-Error is Blue.')
            plt.plot(steps, reward_history, 'r',
                     steps, ev_history, 'g',
                     steps, td_error_history, 'b')
            plt.figure('Anomaly Graph')
            plt.title('Anomaly is Green, Unpredictable input is Red.')
            plt.plot(steps, anomaly_history, 'g',
                     steps, anomalous_input_history, 'r')
            plt.show()

        return fitness


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_parameters', action='store_true')
    parser.add_argument('--best_parameters',    action='store_true')
    parser.add_argument('--file', type=str,   default='checkpoint')
    parser.add_argument('-c', '--cycles',     type=int, default=10000)
    parser.add_argument('-p', '--population', type=int, default=100)
    parser.add_argument('-n', '--processes',  type=int, default=4)
    parser.add_argument('--profile',          action='store_true')
    args = parser.parse_args()

    BG_Test.cycles = args.cycles

    if args.default_parameters or args.best_parameters:
        BG_Test.debug = True

        if args.default_parameters:
            indv = BG_Test()
            print("Default Parameters")
        elif args.best_parameters:
            indv = genetics.Population(args.file, 1)[0]
            print("Best of population")

        print(indv)
        print()
        print("Evaluate returned", indv.evaluate())
    else:
        population = genetics.Population(args.file, args.population)
        genetics.evolutionary_algorithm(
            BG_Test,
            population,
            mutation_probability            = 0.50,
            mutation_percent                = 0.50,
            num_processes                   = args.processes,
            profile                         = args.profile,)
