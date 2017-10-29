#!/usr/bin/python3
"""
Written by David McDougall, 2017

This experiment combines an eye sensor, a spatial pooler, and a statistical
classifier to recognise images.  The primary purpose of this is to examine the
eye sensor - spatial pooler complex in isolation.  If it is successful it will
serve as a unit test for further spatial pooler experimentation.
"""

import htm
import genetics
import numpy as np
import math
import scipy.ndimage
import random
import matplotlib.pyplot as plt
import sys
from sdr import SDR
from datasets import Dataset


class RecognitionExperiment(genetics.Individual):
    debug          = False
    image_cycles   = None   # Set by main function.
    time_per_image = None   # Set by main function.

    parameters = [
        'eye',
        'sp',
        'columns',
        'radii',
        'sdrc',
    ]
    fitness_names_and_weights = {
        'score':        +1,
        'baseline':     -1,
        'time':         -5,
        'memory':       -5,
    }

    def __init__(self,):
        # These are the initial parameters which the seed population is based on.
        self.columns = (100, 100)
        self.radii = (3, 3)
        self.sp = htm.SpatialPoolerParameters(
                boosting_alpha    = 0.001138,
                permanence_dec    = 0.01057,
                permanence_inc    = 0.03549,
                permanence_thresh = 0.279,
                potential_pool    = 914.713456247026,
                sparsity          = 0.014846076410960425,)
        self.sdrc = htm.SDRC_Parameters(alpha=1/1000)
        self.eye = htm.EyeSensorParameters(
            angle_encoder = htm.RandomDistributedScalarEncoderParameters(
                resolution = 0.030557369853445093,
                size       = 158.533689097986,
                sparsity   = 0.15,),
            angle_stddev = 0.3146436177830795,
            angular_velocity_encoder = htm.RandomDistributedScalarEncoderParameters(
                resolution = 0.03620871141979454,
                size       = 164.34134314569107,
                sparsity   = 0.155290883411185,),
            edge_encoder = htm.ChannelThresholderParameters(
                channel = htm.ChannelEncoderParameters(
                    num_samples = 3.333073696295324,
                    sparsity    = 0.1766435461176121,),
                mean = 0.4460895680384542,
                stddev = 0.15968979856796287,),
            eye_dimensions = (100, 100),
            fovea_param_1 = 0.017339086018333914,
            fovea_param_2 = 15.969744316723455,
            hue_encoder = htm.ChannelEncoderParameters(
                num_samples = 5.437913467852119,
                sparsity    = 0.1783,),
            max_scale = 4.901,
            min_scale = 0.9358,
            num_cv = 599.6810879479867,
            pos_stddev = 1.0373833293514578,
            position_encoder = htm.RandomDistributedScalarEncoderParameters(
                resolution = 1,
                size       = 128,
                sparsity   = 0.15781491121807742,),
            sat_encoder = htm.ChannelEncoderParameters(
                num_samples = 3.936766985163579,
                sparsity    = 0.1744,),
            scale_encoder = htm.RandomDistributedScalarEncoderParameters(
                resolution = 0.2,
                size       = 110.31163074891285,
                sparsity   = 0.13087563171844932,),
            scale_stddev = 2,
            scale_velocity_encoder = htm.RandomDistributedScalarEncoderParameters(
                resolution = 0.2,
                size       = 173.79805214927734,
                sparsity   = 0.2512633236951939,),
            val_encoder = htm.ChannelEncoderParameters(
                num_samples = 4.961,
                sparsity    = 0.2018,),
            velocity_encoder = htm.RandomDistributedScalarEncoderParameters(
                resolution = 0.8181249205408995,
                size       = 128,
                sparsity   = 0.15,),)

    def evaluate(self):
        data        = Dataset('datasets/textures/')
        max_dist    = -32
        timer       = genetics.speed_fitness(threshold = 60, maximum = 2*60)
        sensor      = htm.EyeSensor(self.eye)
        sp          = htm.SpatialPooler(self.sp,
                                        input_sdr   = sensor.optic_sdr,
                                        column_sdr  = SDR(self.columns),
                                        radii       = self.radii,
                                        multisegment_experiment = None)
        classifier  = htm.SDR_Classifier(self.sdrc,
                                        sp.columns,
                                        (len(data.names),),
                                        output_type='pdf')
        baseline    = htm.RandomOutputClassifier((len(data.names),))

        time_per_image   = int(round(self.time_per_image))
        num_images       = int(round(self.image_cycles))
        if self.debug:
            sampler      = htm.EyeSensorSampler(sensor, num_images, 30)
            memory_perf  = 0    # Debug takes extra memory, this is disabled during debug.
        sp_score         = 0
        baseline_score   = 0
        # Outer loop through images.
        for img_num in range(num_images):
            # Setup for a new image
            data.random_image()
            sensor.new_image(data.current_image)
            # Determine where the eye will look on the image.
            positions = data.points_near_label(max_dist=max_dist, number=time_per_image)

            # Inner loop through samples from each image.
            for sample_point in positions:
                # Classify the image.
                sensor.randomize_view()     # Get a new orientation and scale.
                sensor.position = sample_point
                sp.compute(input_sdr=sensor.view())
                sp_prediction       = classifier.predict(sp.columns)
                baseline_prediction = baseline.predict()

                # Compare results to labels.
                label_sample_points = sensor.input_space_sample_points(20)
                labels              = data.sample_labels(label_sample_points)
                sp_score       += data.compare_label_samples(sp_prediction, labels)
                baseline_score += data.compare_label_samples(baseline_prediction, labels)
                # Learn.
                sp.learn()
                classifier.train(sp.columns.index, labels)
                baseline.train(labels)

            # Sample memory usage.
            if img_num == min(10, num_images-1) and not self.debug:
                # I want each process to take no more than 20% of my RAM or 2.5
                # GB.  This should let me run 4 processes + linux + firefox.
                memory_perf = genetics.memory_fitness(2e9, 4e9)

        sp_score        /= sp.age
        baseline_score  /= sp.age
        time_perf = timer.done()    # Stop the timer before debug opens windows and returns control to user.
        if self.debug:
            print(sp.statistics())
            sampler.view_samples()

        return {
            'baseline':  baseline_score,
            'score':     sp_score,
            'time':      time_perf,
            'memory':    memory_perf,
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', type=int, default=1000,
        help='Number of images to train on.')
    parser.add_argument('-t', '--time', type=int, default=10,
        help='How long to look at each image.')
    parser.add_argument('--default_parameters', action='store_true')
    parser.add_argument('--best_parameters', action='store_true')
    parser.add_argument('--file', type=str, default='checkpoint')
    parser.add_argument('-p', '--population', type=int, default=100)
    parser.add_argument('-n', '--processes', type=int, default=6)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    RecognitionExperiment.image_cycles   = args.images
    RecognitionExperiment.time_per_image = args.time

    if args.default_parameters or args.best_parameters:
        RecognitionExperiment.debug = True

        if args.default_parameters:
            indv = RecognitionExperiment()
            print("Default Parameters")
        elif args.best_parameters:
            indv = genetics.Population(args.file, 1)[0]
            print("Best of population")

        print(indv)
        print("Evaluate returned", indv.evaluate())
    else:
        population = genetics.Population(args.file, args.population)
        genetics.evolutionary_algorithm(
            RecognitionExperiment,
            population,
            mutation_probability            = 0.50,
            mutation_percent                = 0.50,
            num_processes                   = args.processes,
            profile                         = args.profile,)
