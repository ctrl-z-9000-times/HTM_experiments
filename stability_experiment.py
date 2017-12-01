#!/usr/bin/python3
# Written by David McDougall, 2017
# See file: htm_stability.odt

import numpy as np
from htm import *
import genetics
import itertools
import random
import math

object_size = 10
num_objects = 100
def object_dataset():
    alphabet = [chr(ord('A') + x) for x in range(26)]
    inputs   = [''.join(chrs) for chrs in itertools.product(alphabet, repeat=3)]
    objects  = [random.sample(inputs, object_size) for x in range(num_objects)]
    return inputs, objects

class StabilityExperiment(genetics.Individual):
    parameters = [
        'alpha',
        'cols',
        'col_sparsity',
        'dec',
        'enc_size',
        'enc_sparsity',
        'inc',
        'min_stab',
        'pp',
        'prox_segs',
        'thresh',
        'init_dist',
    ]
    fitness_names_and_weights = {
        'score':  +2,
        'ovlp':   +1,
        'memory': -0.5,}

    def __init__(self):
        # Updated 11/30/2017, file: test5
        self.alpha        = 0.00694063418348
        self.col_sparsity = 0.01 # 0.00579464350062
        self.cols         = 2000 # 1001.03540425
        self.dec          = 0.00357969954117
        self.enc_size     = 1950.36653676
        self.enc_sparsity = 0.0170122738181
        self.inc          = 0.0620456340463
        self.init_dist    = (0.049742999835167091, 0.032157580415712081)
        self.min_stab     = 0.642926775309
        self.pp           = 1393.9014024
        self.prox_segs    = 7.31731263821
        self.thresh       = 0.138191824662

    def evaluate(self, debug):
        # SETUP
        inputs, objects = object_dataset()
        enc = EnumEncoder(self.enc_size, self.enc_sparsity, diag=False)
        enc.output_sdr = SDR(enc.output_sdr, average_overlap_alpha=self.alpha)
        sp = SpatialPooler(
            SpatialPoolerParameters(
                permanence_inc      = self.inc,
                permanence_dec      = self.dec,
                permanence_thresh   = self.thresh,
                potential_pool      = self.pp,
                sparsity            = self.col_sparsity,
                boosting_alpha      = self.alpha,),
            input_sdr  = SDR(enc.output_sdr, activation_frequency_alpha=self.alpha),
            column_sdr = SDR((self.cols,)),
            multisegment_experiment=self.prox_segs,
            init_dist  = self.init_dist,)
        sdrc = SDR_Classifier(SDRC_Parameters(alpha=0.001),
                input_sdr    = sp.columns,
                output_shape = (num_objects,),
                output_type  = 'index')

        def measure():
            # Compute every sensation for every object.
            objects_columns = []
            for obj in objects:
                objects_columns.append([])
                for sensation in obj:
                    enc.encode(sensation)
                    sp.compute(input_sdr=enc.output_sdr)
                    objects_columns[-1].append(SDR(sp.columns))

            # Measure classification accuracy.
            score = 0
            max_score = 0
            for object_id, obj_cols in enumerate(objects_columns):
                for sp_cols in obj_cols:
                    prediction = np.argmax(sdrc.predict(sp_cols))
                    if prediction == object_id:
                        score += 1
                    max_score += 1
            score = score / max_score

            # Measure column overlap within objects.
            overlap_stability = 0
            comparisions = 0
            for obj_cols in objects_columns:
                for c1, c2 in itertools.combinations(obj_cols, 2):
                    overlap_stability += c1.overlap(c2)
                    comparisions      += 1
            stability = overlap_stability / comparisions

            # Measure column overlap between objects.
            overlap_between_objects = 0
            comparisions = 0
            skip_compare = itertools.cycle([False] + [True] * 24)
            for obj1_cols, obj2_cols in itertools.combinations(objects_columns, 2):
                for c1 in obj1_cols:
                    for c2 in obj2_cols:
                        if next(skip_compare):
                            continue
                        overlap_between_objects += c1.overlap(c2)
                        comparisions += 1
            distinctiveness = overlap_between_objects / comparisions
            return stability, distinctiveness, score

        if debug:
            untrained_stability, untrained_distinctiveness, untrained_score = measure()
            print('NUM-OBJ ', num_objects)
            print("OBJ-SIZE", object_size)
            print(sp.statistics())
            print('UNTRAINED INTER OBJECT OVERLAP', untrained_distinctiveness)
            print('UNTRAINED INTRA OBJECT OVERLAP', untrained_stability)
            print('UNTRAINED SCORE', untrained_score)
            print('Training ...')

        # RUN ONLINE
        prev_cols = SDR(sp.columns); prev_cols.zero()
        for step in range(num_objects * (object_size ** 2) * 10):
            if step % 2*object_size == 0:
                object_id = random.choice(range(len(objects)))
            sensation = random.choice(objects[object_id])
            enc.encode(sensation)
            sp.compute(input_sdr=enc.output_sdr)
            sp.stabilize(prev_cols, self.min_stab)
            prev_cols = SDR(sp.columns)
            sp.learn()
            sdrc.train(sp.columns, (object_id,))

        trained_stability, trained_distinctiveness, trained_score = measure()

        if debug:
            print('TRAINED FOR CYCLES', step+1)
            print('TRAINED INTER OBJECT OVERLAP', trained_distinctiveness)
            print('TRAINED INTRA OBJECT OVERLAP', trained_stability)
            print('TRAINED SCORE',   trained_score)
            print()
            print('Encoder Output', enc.output_sdr.statistics())
            print(sp.statistics())

        return {
            'score':  trained_score,
            'ovlp':   trained_stability / trained_distinctiveness,
            'memory': genetics.memory_fitness(4e9, 5e9),
        }

if __name__ == '__main__':
    genetics.ExperimentMain(StabilityExperiment)
