#!/usr/bin/python3
# Written by David McDougall, 2017

import htm
import genetics
import numpy as np
import math
import scipy.ndimage
import random
import matplotlib.pyplot as plt
import sys
import time
from datasets import Dataset

"""
Outstanding tasks:

0)  Genetic algorithms
    .4) Recognition experiment, should show some improvements over time...
            It did! It improved but then I changed all of my code.  I should 
            reproduce this AFTER my code and datasets are fixed up.
    .6)  gitHUB

STOP    Make recognition experiment work reasonably well with new texture
        dataset before continuing.  This will likely involve a time consuming
        parameter space search...


I think the next steps should building a temporal pooler, which will allow me to
combine many (an arbitrarily large number of) spatial pooler outputs taken over
time into a single coherent model of the world.  Before I can begin building my
temporal pooler I need several other things done.

1)  My spatial pooler should be rock solid.
    I want to have a suite of internal diagnostics such as:
        * Activation frequency statistics, boost-factor statistics, and entropy.
        * Stability, short and long term w/ randomly sampled inputs.
        * Noise Robustness, this should capture its inputs in the same way that
          stability monitoring randomly samples from the input stream, maybe
          even use the same samples?
        * These diagnostics should run themselves, periodically writing/logging
          their findings to a configurable location.
    I want it to run reasonably fast.
        No parallelism, the parameter search needs all of the cores
        * Cython?  It's not really needed for my long term plan but it could be
          fun to learn about and use, and it could really speed the program up.
        !!! Cython rocks.  In a few hours I was able to get a 30% speed up
          by getting around numpy's limitations and pythons type system.  
        * I think I could probably get another 5-10% speedup by fixing data
          types, buffer locations, localizing variables, and other non-algorithmic 
          changes.  
        * Look into using a hash set to represent input activations.  In theory
          this trades off memory for computations but I think it might perform
          better in the cache system.  So I have 8MB cache and a quick estimate
          shows easily 5MB to store a reasonably sized hash set which is
          uncomfortably close to the 8MB limit, which BTW is shared between the
          two threads on the core and might even be shared by the cores.  And if
          the hashset doesn't fit in cache then there really isn't an advantage
          to using it.  
            size (in ON bits):   eye-shape * depth * 20% = 200k active bits,
            Hash table array:    size * 6 (bool w/ 5/6 sparsity),
            Actual values array: size * 4 (int32),
            safety margin:       2x
            total size:          2-4 MB


2)  My encoders should be rock solid too. They really aren't as interesting but 
    there is at least one task left to do with them which should be finished 
    before I start in on the next chapter.  
    TODO)  Measure actual encoder sparsities, show [min, mean, std, max]
            Channel Encoder,
            Threshold Encoder,
            EyeSensor
                I'm not going to do this task until I need to care about the channel
                encoder parameters, which I don't yet.  Hopefully the GA will tune
                these parameters for me...

3)  I should review the literature on all of this, but especially temporal
    poolers.  Read the chapters from J. Hawkins textbook on the subject.  

4)  I need timeseries datasets to test against.
    * Visual imagery with a constant motion such as 3 degree rotation between 
      each input frame or a 3 pixel movement in one direction.  The goal is to
      show that by using a temporal pooler, I get better accuracy than is 
      possible by combining multiple SP+SDRC predictions. 
    * Scalar timeseries - These are the easiest to find on the web and they're
      usually smaller so they run fast.
    I need at least one small and fast dataset to verify the implementation
    works and several more difficult datasets to show that it works well.


14)  Make, integrate, and test RL motor controls.
        TODO: Show a rolling exponential average reward, average last 5~10 episodes?
        TODO: Add a motion sensor to the EyeSensor

15)  Make SP controls, train on the WORKING RL controls.  
"""


"""
Texture Dataset Best Runs:

Another more promising run, currently comparing the old and new encoders.
    Channel Encoder (256, 256, 3) -> (256, 256, 3, 6)
        Sparsity 0.167   
    Eye Parameters
        Retina Input -> Output Shapes (256, 256) -> (256, 256, 20)
        Motor Sensors 1024, Motor Controls 600
    Proximal Synapse Parameters
        Input -> Output shapes are (256, 256, 20) -> (128, 128)
        Coincidence Ratio 0.1 / 0.02 = 5.0
        Density within 1/2/3 deviations: 0.539 / 0.208 / 0.0357
        Radii (6, 6)    Num Inputs 3072
    Proximal Synapse Parameters
        Input -> Output shapes are (128, 128) -> (64, 64)
        Coincidence Ratio 0.1 / 0.02 = 5.0
        Density within 1/2/3 deviations: 0.862 / 0.442 / 0.0672
        Radii (16, 16)  Num Inputs 2048
    Random Output Classifier
        Input -> Output shapes are (20480,) -> (11,)
    Training for 500 cycles, 10 per image
    Testing for 500 cycles, 10 per image
    Baseline Score: 54.29792 / 500 = 0.10859584
    Score: 118.943414883 / 500 = 0.237886829765
    Score X: 127.744448492 / 500 = 0.255488896984


Larger encoders == improved accuracy
Old and new encoders at same performace.
    Channel Encoder (256, 256, 4) -> (256, 256, 4, 10)
        Sparsity 0.2   
    Eye Parameters
        Retina Input -> Output Shapes (256, 256) -> (256, 256, 42)
    Proximal Synapse Parameters
        Input -> Output shapes are (256, 256, 42) -> (128, 128)
        Coincidence Ratio 0.1 / 0.02 = 5.0
        Density within 1/2/3 deviations: 0.601 / 0.236 / 0.0413
        Radii (4.5, 4.5)    Num Inputs 4096
    Proximal Synapse Parameters
        Input -> Output shapes are (128, 128) -> (64, 64)
        Coincidence Ratio 0.1 / 0.02 = 5.0
        Density within 1/2/3 deviations: 0.862 / 0.442 / 0.0672
        Radii (16, 16)  Num Inputs 2048
    SDR Classifier alpha 0.001
        Input -> Output shapes are (20480,) -> (11,)
    Training for 500 cycles, 10 per image
    ...
    Baseline Score: 50.506 / 500 = 0.101012
    Score: 147.749063682 / 500 = 0.295498127364
    Score X: 150.466632152 / 500 = 0.300933264303
    Elapsed Time 30:19 sec
"""


def texture_dataset_preview():
    print("Texture Dataset Preview")
    data    = Dataset('datasets/textures/')
    if False:
        train_data, test_data = data.split_dataset(75, 25)
        print("Training Data Statistics")
        print(train_data.statistics())
        print()
        print("Testing Data Statistics")
        print(test_data.statistics())
        print()
    sensor  = EyeSensor()
    sensor.max_scale = 1
    sensor.max_scale = 2.5
    t1 = 10     # Number of images to sample from
    t2 = 5      # Number of potential samples from each image.
    while True:
        sampler = EyeSensorSampler(sensor, t1*t2)
        for i1 in range(t1):
            data.random_image()
            print(data.current_image)
            view_points = data.points_near_label(max_dist=-32, number=t2)
            for i2 in view_points:
                sensor.new_image(data.current_image)
                sensor.randomize_view() # Random scale & orientation
                sensor.position = i2
                sensor.view()
        sampler.view_samples()
# texture_dataset_preview()


class RecognitionExperimentParameters(genetics.Individual):
    parameters = [
        'eye',
        'sp',
        'sdrc',
        'image_cycles',   
    ]
    fitness_names_and_weights = {
        'recognition':  +1,
        'baseline':     -1,
        'time':          0,
        'memory':        0,
    }
    def __init__(self,):
        # These are the initial parameters which the seed population is based on.
        self.eye = htm.EyeSensorParameters()
        self.sp  = htm.SpatialPoolerParameters(
            column_dimensions = (100, 100),
            radii             = (8, 8),
            potential_pool    = 1024,
        )
        self.sdrc = htm.SDRC_Parameters()
        self.image_cycles = 100


def evaluate_recognition(parameters, debug=False):
    data = Dataset('datasets/textures/')
    train_data, test_data = data.split_dataset(75, 25, verbosity=0)
    max_dist = -32

    timer       = genetics.speed_fitness(threshold = 30, maximum = 2*60)
    eye_params  = parameters.eye
    sp_params   = parameters.sp
    sdrc_params = parameters.sdrc
    sensor      = htm.EyeSensor(eye_params)
    sp          = htm.SpatialPooler(sp_params, sensor.view_shape)
    classifier  = htm.SDR_Classifier(sdrc_params, sp.column_dimensions, (len(data.names),), output_type='pdf')
    baseline_classifier = htm.RandomOutputClassifier((len(data.names),))

    time_per_image = 10
    training_time  = int(round(time_per_image * parameters.image_cycles))
    if debug:
        training_time = time_per_image * 2
    for i in range(training_time):
        # Setup for a new image
        if i % time_per_image == 0:
            train_data.random_image()
            sensor.new_image(train_data.current_image)
            eye_positions = train_data.points_near_label(
                                    max_dist = max_dist,
                                    number=time_per_image)
        # Setup and compute for this cycle
        sensor.randomize_view()     # Random scale and orientation
        sensor.position = eye_positions.pop()
        labels_pdf = train_data.sample_labels(sensor.input_space_sample_points())
        view       = sensor.view()
        columns    = sp.compute(view)
        classifier.train(columns, labels_pdf)
        baseline_classifier.train(labels_pdf)
        # Sample the memory usage.
        if i == time_per_image-1:
            memory_performance = genetics.memory_fitness([sensor, sp, classifier],  0.5e9, 2e9)

    testing_time     = min(1000, time_per_image * 100)
    if debug:
        testing_time = time_per_image * 2
    score            = 0    # Regular accuracy, summed results of dataset.compare_label_samples
    baseline_score   = 0
    for i in range(testing_time):
        # Setup for a new image
        if i % time_per_image == 0:
            test_data.random_image()
            sensor.new_image(test_data.current_image)
            eye_positions = test_data.points_near_label(
                                    max_dist = max_dist,
                                    number   = time_per_image)
        # Setup and compute for this cycle
        sensor.randomize_view()
        sensor.position = eye_positions.pop()
        view            = sensor.view()
        columns         = sp.compute(view, learn=False)
        prediction      = classifier.predict(columns)
        baseline        = baseline_classifier.predict()
        # Compare results to labels
        labels         = test_data.sample_labels(sensor.input_space_sample_points())
        score          += test_data.compare_label_samples(prediction, labels)
        baseline_score += test_data.compare_label_samples(baseline, labels)
    score           /= testing_time
    baseline_score  /= testing_time

    return {
        'recognition':  score,
        'baseline':     baseline_score,
        'time':         timer.done(),
        'memory':       memory_performance,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='checkpoint')
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('-p', '--population', type=int, default=50)
    parser.add_argument('-n', '--processes', type=int, default=7)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()
    if args.debug:
        if args.seed:
            indv = RecognitionExperimentParameters()
            print("Seed Parameters")
        else:
            pop, gen = genetics.checkpoint([], 0, filename=args.file)
            indv = pop[0]
            print("Best of Generation %d"%gen)
            print("Fitness:", indv.fitness)
        print(indv)
        retval = evaluate_recognition(indv, debug=args.debug)
        print("evaluate_recognition returned", retval)
    else:
        pop, gen = genetics.genetic_algorithm(
            RecognitionExperimentParameters, evaluate_recognition,
            population_size                 = args.population,
            num_epochs                      = 9999,
            seed                            = args.seed,
            seed_mutations_per_parameter    = 1,
            seed_mutation_percent           = 0.50,
            mutation_probability            = 1/6,
            mutation_percent                = 0.25,
            filename                        = args.file,
            num_processes                   = args.processes,
            profile                         = args.profile,)
    exit(0)


class NStepQueue:
    """
    Holds the past N-Steps of experience and applies multi-step bootstrapping
    to all given experiences.

    The advantage of N-Step bootstrapping is that it frees the agent from the
    observation-action  time step.  It allows the agent to look for rewards that
    are more than one step ahead of it. The issue is that the time-granularity
    is not necessarily the same as the amount of time it takes to do anything
    interesting.  For example, the agent could receive input every milisecond
    but the rewards are delayed by a full second after the action which gets
    them.  With N=1000, 1 full second of the agents experiences leading up to
    the reward will include the reward, as  opposed to with N=0 only the
    milisecond before the reward includes it.  This allows the agent to receive
    input at an arbitrary rate w/r/t time and continue learning based on a
    controlled time window for receiving rewards.
    """
    class Experience:
        """
        A single instance of training data, this class is used by the NStepQueue

        Attributes observation & action are from an instantaneous sample taken
                   durring training.

        Attribute 'n' is how many steps in the past this experience was created.

        Attribute next_observation was received N+1 steps after observation was.
        Attribute next_discount is the future-discount to apply to 
                  next_observation's reward.

        Attribute reward is the time discounted sum of rewards received in the 
                  N+1 steps after taking the action.
        """
        def __init__(self, observation, action, reward):
            self.observation        = observation
            self.action             = action
            self.n                  = None
            self.next_observation   = None
            self.next_discount      = None
            self.reward             = reward

    def __init__(self, n_step, future_discount, output_callback):
        """
        Argument 'n_step'
            This is the N-Step of the TD(N-Steps) method.
            How many steps to look forward for the reward.
            Basic Q-Learning uses N=0
            MonteCarlo Methods use N=infinity

        Argument future_discount...

        Argument 'output_callback' is called with an experience every time an
            experience leaves this queue.  
        """
        assert(n_step >= 0)
        assert(0.0 <= future_discount and future_discount <= 1.0)
        self.n_step         = n_step
        self.discount       = future_discount
        self.experience_q   = []
        self.output         = output_callback

    def add(self, observations, actions, reward=None):
        """
        Adds half of an experience.

        The result of this experience does not need to be known.
            reward is optional, if not given it must be updated before the next observation is added.
            The Experience.next_obs will be filled in by this class when it is received.
        """
        assert(not self.experience_q or self.experience_q[-1].reward is not None)
        expr = NStepQueue.Experience(observations, actions, reward)
        self.experience_q.append(expr)
        if len(self.experience_q) > self.n_step + 1:
            self._calc()

    def update_reward(self, reward):
        """
        Assigns a reward to the most recently added observation-action pair.
        This is useful because actions' rewards are not known until the
        following time step.
        """
        if self.experience_q:
            self.experience_q[-1].reward = reward

    def reset(self):
        """Call this after any discontinuity in the input!"""
        while self.experience_q:
            self._calc()

    def _calc(self):
        """Process q[0]"""
        if not self.experience_q:
            return              # No data, nothing to do.
        if len(self.experience_q) == 1:
            # Can't learn from a single sample of data, no reward signal to
            # learn from.  
            self.experience_q.pop()   # Discard trailing experience
            return

        # Find the time discounted rewards
        reward = 0
        for expr in reversed(self.experience_q):
            if expr.reward is None:
                # The latest experience hasn't received a reward and the last 
                # experience before a reset never will.  Those experiences'
                # rewards are estimated by the AI instead.  
                continue
            reward = expr.reward + reward * self.discount

        # Finish this sample of experience.
        expr                  = self.experience_q.pop(0)
        expr.n                = len(self.experience_q)
        expr.reward           = reward
        expr.next_observation = self.experience_q[-1].observation
        expr.next_discount    = self.discount ** expr.n     # (N+1) ???
        self.output(expr)


class SupervisedController:
    """
    Supervised learning for eye saccades.
    This uses a table to learn the expected values of each state-action pair.
    The table is updated using the TD-error method.
    """
    def __init__(self, input_shape, output_shape, output_sparsity=.05):
        """
        """
        self.learning_rate   = 1/100
        self.input_shape     = tuple(input_shape)
        self.output_shape    = tuple(output_shape)
        self.input_size      = np.product(self.input_shape)
        self.output_size     = np.product(self.output_shape)
        self.on_bits         = max(1, int(round(output_sparsity * self.output_size)))
        self.xp_q            = NStepQueue(3, .90, self.learn)
        self.expected_values = np.random.random((self.input_size, self.output_size)) * self.learning_rate
        self.expected_values = np.array(self.expected_values, dtype=np.float32)
        print("Supervised Controller")
        print("\tExpected Values shape:", self.expected_values.shape)
        print("\tFuture discount:", self.xp_q.discount)
        print("\tLearning Rate:", self.learning_rate)

    def reset(self):
        self.xp_q.reset()

    def best_action(self, observation):
        """Returns the suggested control vector."""
        action_values = self.expected_values[observation]
        action_values = np.sum(action_values, axis=tuple(range(len(self.input_shape))), keepdims=False)
        k = self.on_bits
        actions = np.argpartition(-action_values, k-1)[:k]
        return actions

    def learn(self, experience):
        xp = experience
        # Use xp.next_observation to estimate the value of xp.action after 
        # observing the result of xp.action for xp.n steps.
        next_action = self.best_action(xp.next_observation)
        next_value = np.mean(self.expected_values[xp.next_observation, :][..., next_action])

        # Find the value of xp's state & action by adding the rewards which
        # accumulated in the past N steps (xp.reward) with the expected value of
        # xp.next_observation.  The future discount is also applied.
        true_value = xp.reward + next_value * xp.next_discount

        # Adjust the expected values table to match the new estimate of this
        # state & action's expected value.
        update = true_value - self.expected_values[xp.observation, :][..., xp.action]
        update *= self.learning_rate
        self.expected_values[xp.observation, :][..., xp.action] = update

    def act(self, observation, reward):
        """
        Interact with and learn from the environment.
        Returns the suggested control vector.
        """
        observation = np.ravel_multi_index(observation, self.input_shape)
        self.xp_q.update_reward(reward)
        action = self.best_action(observation)
        self.xp_q.add(observation, action)
        action = np.unravel_index(action, self.output_shape)
        return action


class EyeClassifier:
    """Classifies an image using an EyeSensor and several SpatialPoolers."""
    def __init__(self, dataset, diag=False):
        """
        Argument dataset, this class uses the datasets current image and label.
        """
        self.age = 0
        self.dataset = dataset
        self.encoder = EyeSensor()

        from hierarchial_spatial_pooler import HierarchialSpatialPooler
        self.vision = HierarchialSpatialPooler(
            {   'input_dimensions'  : self.encoder.view_shape,
                'column_dimensions' : (128, 128),
                'radii'             : (3, 3),
            },
            {   'column_dimensions' : (64, 64),
                'radii'             : (16, 16),
            },
        )

        label_shape = (len(self.dataset.names),)
        self.classifier = htm.SDR_Classifier(self.vision.column_dimensions, label_shape, 'pdf')

        self.motor = htm.SpatialPooler(
            input_dimensions  = self.encoder.motor_shape,
            column_dimensions = (int(self.encoder.motor_shape[0] / 2),),
        )

        from hierarchial_spatial_pooler import SDR_Concatenator
        self.control_input_joiner = SDR_Concatenator(self.vision.column_dimensions, self.motor.column_dimensions)

        self.sp_controller = htm.SpatialPooler(self.control_input_joiner.output_dimensions,
                                               self.encoder.control_shape,)

        self.rl_controller = SupervisedController(self.control_input_joiner.output_dimensions,
                                                  self.encoder.control_shape,)

    def reset(self):
        self.encoder.reset()
        self.rl_controller.reset()

    def train(self, learn=True, time_limit=20, verbosity=1):
        """Trains on the current image in the dataset."""
        self.controller = 'rl'
        total_rewards = 0
        reward = 0
        self.encoder.new_image(self.dataset.current_image)
        for step in range(time_limit):
            prediction, action = self.tick(learn=True, reward=reward)
            self.encoder.move(action)
            reward = self.reward_function(prediction)
            total_rewards += reward
        print("total_rewards", total_rewards)

    def classify(self, *args):
        1/0 # Unimplemented

    def tick(self, learn=True, reward=None):
        """
        Process the current sensory input and output predictions and actions.

        Returns pair of (prediction, action)
            Where prediction is an array of probabilities, in parallel with each
                label in self.dataset.sorted_names.
            Where action is index array of control vectors.
        """
        self.age += 1
        visual_input  = self.encoder.view(diag=False)
        visual_state  = self.vision.compute(visual_input, learn=learn)
        prediction    = self.classifier.predict(visual_state)

        if learn:
            sample_points = self.encoder.input_space_sample_points()
            labels        = self.dataset.sample_labels(sample_points)
            self.classifier.train(visual_state, labels)

        motor_input = self.encoder.motor_sdr
        motor_state = self.motor.compute(motor_input, learn=learn)
        control_input = self.control_input_joiner.compute(visual_state, motor_state)

        rl_cv = self.rl_controller.act(control_input, reward)

        # Have the SP controller learn from the RL controller
        dont_care = self.sp_controller.proximal.compute(control_input)
        self.sp_controller.proximal.learn(rl_cv)

        return prediction, rl_cv

    def reward_function(self, prediction):
        """
        Argument self.dataset must have the current image loaded
        Argument prediction ...

        There are 2 different reward signals which are summed.

        1) Positioning the eye sensor over a label (small reward)
        2) Correctly identifying a label (larger reward)

        Rewards are finite and diminish after each time they are given.  The purpose
        is to keep the eye looking for new rewards.  
                TODO: Unimplemented...
        """
        reward = 0
        # Determine what the eye sensor is actually looking at.
        labels = self.dataset.sample_labels(self.encoder.input_space_sample_points())
        labels = labels / np.sum(labels)    # Is a PDF.
        
        # Reward for looking at labeled data.
        pct_unlabeled = labels[self.dataset.sorted_names.index('unlabeled')]
        if pct_unlabeled < .66:
            reward += 1

        # Reward for correctly identifying objects.
        pct_correct = self.dataset.compare_label_samples(prediction, labels)
        if pct_correct > .25:
            reward += 5

        # Penalty for looking at the same spot for too long.
        pass

        return reward


def control_experiment():
    """
    I really don't know what works here so I'm going to start building this
    from the ground up.  Write the experiment and  training functions first
    because they are testable and really should work,  Don't add AI's until
    their test framework works.
    """
    data = Dataset('datasets/small_items/') 
    data.discard_unlabeled_data()
    train_data, test_data = data.split_dataset(75, 25)
    gaze = ai.encoder.gaze_tracking()

