"""
Evolutionary algorithms and supporting tools.

Written by David McDougall, 2017
"""

import numpy as np
import random
import multiprocessing as mp
import cProfile, pstats, tempfile
import traceback
import sys
import time
import signal
import os, os.path
import pickle
import copy
import csv

class Parameters:
    """
    Abstract parent class for genetic material.  Child classes represent
    parameters for a specific type of object.  The purpose of this class is to
    facilitate a parameter search, for example an evolutionary algorithm or
    swarming.

    Class attribute parameters is the list of attributes which make up the 
    mutable parameters, all other attributes are ignored by this class and its
    methods.  The type of these attributes must be one of:
        1) Floating point number
        2) Tuple of floating point numbers
        3) Boolean number
        4) Instance of Parameters or a child class.
    """
    parameters = []

    def __init__(self, **kw_args):
        """
        Child classes override this as needed.  This writes all keyword
        arguments into the new instance as its initial values.
        """
        # This allows child classes to pass their locals() as **kw_args, which
        # is useful for mixing default parameters and **kw_args.   Double
        # underscores are magic and come and go as they please.  Filter them all
        # out.
        dunder  = lambda name: name.startswith('__') and name.endswith('__')
        kw_args = {k:v for k,v in kw_args.items() if not dunder(k)}

        p_set = set(self.parameters)
        for attr, value in kw_args.items():
            if attr not in p_set:
                raise TypeError("'%s' is not a valid keyword argument to this class"%attr)
            setattr(self, attr, value)
        missing_parameters = [attr for attr in self.parameters if not hasattr(self, attr)]
        if missing_parameters:
            raise TypeError("%s.__init__() missing parameters: %s"%(type(self).__name__, ', '.join(missing_parameters)))

    def mutate(self, percent):
        """
        Randomly change a single parameter by at most the given percent.
        This uses an exponential mutation rate: (1 + percent) ^ uniform(-1, 1)
        """
        def attr_probability(attr):
            """This assigns all attributes uniform probability of mutation."""
            value = getattr(self, attr)
            if isinstance(value, Parameters):
                return len(value)
            elif isinstance(value, tuple):
                return len(value)
            elif value is None:
                return 0    # Don't try to mutate attributes which are set to None.
            else:
                # Floats and Bools
                return 1
        # Pick a parameter to mutate and get its value.
        probabilities   = [attr_probability(attr) for attr in self.parameters]
        probabilities   = np.array(probabilities) / np.sum(probabilities)
        param_idx       = np.random.choice(len(self.parameters), p=probabilities)
        param           = self.parameters[param_idx]
        value           = getattr(self, param)
        if isinstance(value, Parameters):
            value.mutate(percent)
        elif isinstance(value, bool):
            setattr(self, param, not value)
        else:
            # modifier = (1 - percent) + 2 * percent * random.random()
            modifier = (1 + abs(percent)) ** (random.random()*2-1)
            if isinstance(value, tuple):
                # Mutate a random element of tuple
                index    = random.randrange(len(value)) # Pick an element to mutate
                new_elem = value[index] * modifier      # All elements are treated as floating point
                # Swap the new element into the tuple.
                new_value = tuple(new_elem if i==index else elem for i, elem in enumerate(value))
                setattr(self, param, new_value)
            else:
                # Mutate a floating point number
                new_value = value * modifier
                if not ((new_value >= 0) == (value >= 0 )):
                    print(value, new_value, percent)
                setattr(self, param, new_value)

    def crossover(self, parents):
        """
        Overwrites all parameters on this instance with the parents parameters.
        Modifies this class *IN PLACE*

        Parent parameters which are missing are not passed down to the child,
        instead the child is left with its default for that parameter.
        """
        types = {type(p).__name__ for p in parents}.union({type(self).__name__}) # Set notation.
        if len(types) != 1:
            print("Warning: crossover applied to different types:", types)
        for attr in self.parameters:
            no_value = object()
            values   = [getattr(obj, attr, no_value) for obj in parents]
            values   = [v for v in values if v is not no_value]
            if not values:
                # All parents are missing this attribute, leave child at default.
                # This happens when a new parameter is added to an existing 
                # population.  The children should get the new parameter and its
                # defaults even though its not inherited from the parent.  
                pass
            elif isinstance(values[0], Parameters):
                getattr(self, attr).crossover(values)
            else:
                if isinstance(values[0], tuple):
                    child_value = tuple(random.choice(elem) for elem in zip(*values))
                else:
                    child_value = random.choice(values)
                setattr(self, attr, child_value)

    def __len__(self):
        accum = 0
        for attr in self.parameters:
            value = getattr(self, attr)
            if isinstance(value, Parameters):
                accum += len(value)
            elif isinstance(value, tuple):
                accum += len(value)
            elif value is None:
                pass
            else:
                accum += 1
        return accum

    def __str__(self):
        s = 'htm.' + type(self).__name__ + "(\n"
        indent      = ' '*4
        attrs       = sorted(self.parameters)
        values      = [str(getattr(self, attr, None)) for attr in attrs]
        inner_block = not any('\n' in v for v in values)
        alignment   = max(len(attr) for attr in attrs)
        for attr, value in zip(attrs, values):
            value = value.replace('\n', '\n' + indent)
            if inner_block:
                padding = ' '*(alignment - len(attr))
            else:
                padding = ''
            s += indent + attr + padding + ' = ' + value + ',\n'
        s = s.rstrip('\n') + ")"
        return s

    @classmethod
    def population_statistics(cls, population):
        """
        Finds the population wide minimum, mean, standard-deviation, and
        maximumn of all parameters.

        Argument population is list of instances of Parameters or a child 
                 class.

        Returns dictionary of {parameter: statistics}
                If parameter refers to a tuple, it is replace with a parameter
                for each element.  If parameter refers to a nested Parameters
                instance then the full path to the parameter is used with a dot
                seperating each parameter name, as in "indv.sensor.encoder.attr".
                Statistics are tuples of (min, mean, std, max).
        """
        table = {}
        if not population:
            return table
        attribute_list = cls.parameters
        for attr in attribute_list:
            no_value   = object()
            data       = [getattr(indv, attr, no_value) for indv in population]
            data       = [v for v in data if v is not no_value]
            data_types = set(type(v) for v in data)
            if int in data_types and float in data_types:
                data_types.remove(int)  # Integers are not part of the spec but are OK.
            if len(data) == 0:
                data_types = [type(None)]   # No data, use data type None.
            if len(data_types) != 1:
                raise TypeError("Not one data type for attribute '%s', found %s"%(attr, data_types))
            data_type  = data_types.pop()
            if issubclass(data_type, Parameters):
                nested = data_type.population_statistics(data)
                for nested_parameter, nested_value in nested.items():
                    table[attr + '.' + nested_parameter] = nested_value
            elif data_type == type(None):
                table[attr] = None
            elif data_type == tuple:
                for index, elem in enumerate(zip(*data)):
                    table[attr + '[%d]'%index] = (
                        np.min(elem),
                        np.mean(elem),
                        np.std(elem),
                        np.max(elem),)
            else:
                table[attr] = (
                    np.min(data),
                    np.mean(data),
                    np.std(data),
                    np.max(data),)
        return table

    # TODO: Move this method to the Population class?
    @classmethod
    def pprint_population_statistics(cls, population, file=sys.stdout):
        """
        Argument file ... default is STDOUT
        Returns the string
        """
        if not population:
            output = "Population is empty."
            if file is not None:
                file.write(output)
            return output
        table = cls.population_statistics(population)
        entries = list(table.items())
        # Push fitness to either the top of the bottom of the table and then
        # sort alphabetically.
        entries.sort(key = lambda entry: (entry[0].count('fitness'), entry[0]))
        lines      = ["Population Statistics for %d %s (%d total)"%(
                    len(population), cls.__name__, len(population.scoreboard))]
        max_name   = max(len(nm) for nm in table.keys()) + 2
        name = 'Parameter'
        lines.append(name+' '+' '*(max_name - len(name))+' Min       | Mean      | Std       | Max')
        for parameter, stats in entries:
            if stats is not None:
                stats = ['%.3e'%st for st in stats]
            else:
                stats = ['None']
            pad = max_name - len(parameter)
            lines.append(parameter+' '+'.'*pad+' '+' | '.join(stats))
        output = '\n'.join(lines) + '\n'
        if file is not None:
            file.write(output)
        return output


class Individual(Parameters):
    """
    Abstract parent class for each member of a population.  Experiments should
    subclass Individual for their main parameter class.

    This classes initializer should accept no arguments.  Use class variables to
    store arguments instead; this works because typically all individuals
    recieve the same arguments, often from the command line.  This makes it
    possible for the Population and evolutionary algorithm to make new
    individuals as needed.

    This class provides a fitness function and several supporting methods. The
    fitness function is a weighted sum; its variables are the attributes of this
    class which are named in fitness_names_and_weights.keys(),
    fitness_names_and_weights[ attrbute-name ] is the weight which is used. For
    example if fitness_names_and_weights = {'accuracy': 1, 'speed': 1/4} then
    self.fitness() would return (self.accuracy * 1 + self.speed * 1/4)

    Attribute filename is assigned when an individual is saved into a 
              population, and is not changed if the individual is saved into
              another population.
    """
    fitness_names_and_weights = {}

    @property
    def fitness(self):
        """
        Returns the weighted sum of fitnesses, as specified in 
                self.fitness_names_and_weights.  If any of the attributes are
                missing or set to None, this returns -infinity.
        """
        # Calculate the current fitness.
        fitness = 0
        for attr, weight in self.fitness_names_and_weights.items():
            if weight == 0:
                continue
            value = getattr(self, attr, None)
            if value is None:
                fitness = float('-inf')
                break
            fitness += value * weight
        return fitness

    def evaluate(self):
        """
        Subclass should use this method to measure the fitness of this set of
        parameters.

        This method is executed in a subprocess to escape the GIL.

        If this method raises an exception then it is assumed this individual is
        at fault and 'self' is discarded from the population.  A stack trace is
        also printed.

        Returns a dictionary which is pickled, sent across the process boundry,
                and passed to self.apply_evaluate_return(...).  It must contain
                the keys of fitness_names_and_weights and its values are the
                fitnesses.  'None' can be used as a placeholder for missing
                values.
        """
        raise TypeError("Abstract Method Called")

    def apply_evaluate_return(self, fitness_dict):
        """
        Convenience method to take the dictionary which the evaluate method
        returns and assign it into attributes on this instance.

        Argument fitness_dict must contain the keys of fitness_names_and_weights
                 and its values are the fitnesses.
        """
        # Check that evaluate_function returned a valid result.
        if set(self.fitness_names_and_weights.keys()) - set(fitness_dict.keys()):
            raise TypeError("Evaluate function returned dictionary which has missing keys.")
        # Put the fitnesses where they belong.
        for attr_name, fitness in fitness_dict.items():
            setattr(self, attr_name, fitness)

    # TODO: Move this method to the Population class?
    @classmethod
    def population_fitness(cls, population):
        """Finds the min/mean/std/max of each component of fitness."""
        stats = {}
        name_n_weights = list(cls.fitness_names_and_weights.items())
        name_n_weights.append(("fitness", None))
        for attr, weight in name_n_weights:
            data = [getattr(indv, attr, None) for indv in population]
            if any(datum is None for datum in data):
                stats[attr] = None
            else:
                stats[attr] = (
                    np.min(data),
                    np.mean(data),
                    np.std(data),
                    np.max(data),)
        return stats

    # TODO: Move this method to the Population class?
    @classmethod
    def population_statistics(cls, population):
        """
        Combines Individual's fitness statistics with the parent classes 
        parameter statistics.
        """
        table = super().population_statistics(population)
        fitness_stats = cls.population_fitness(population)
        fitness_stats = {'fitness.'+attr:stat for attr, stat in fitness_stats.items()}
        table.update(fitness_stats)
        return table

    def __str__(self):
        """Combines the parents parameter table with a fitness table."""
        param_table = super().__str__()
        entries = [(attr, getattr(self, attr, None)) for attr in self.fitness_names_and_weights.keys()]
        entries.sort(key = lambda attr_value: attr_value[0])
        entries.append(("Weighted Fitness", self.fitness))
        entries_w_values = [row is None for row in entries if row is not None]
        if not entries_w_values:
            fitness_table = "Fitness not yet evaluated..."
        elif len(entries_w_values) == 1:
            fitness_table = "Fitness %g"%entries_w_values[0]
        else:
            max_name = max(len(nm) for nm, ft in entries)
            lines = ['fitness']
            for attr, value in entries:
                pad = max_name - len(attr) + 2
                lines.append('  ' + attr +' '+ '.'*pad +' '+ str(value))
            fitness_table = '\n'.join(lines)
        return fitness_table + '\n' + param_table


class Population(list):
    """
    Manages groups of Individuals.

    All populations exist on file, in the directory:
    .   [program_name]_data/[population_name]/
    .       Where program_name is the name of the program (from sys.argv[0]), 
    .           without its extension.
    .       Where population_name is an attribute and the first argument to the
    .           __init__ method.
    This directory contains a file for each Individual and a scoreboard file.
    Low performing Individuals are not removed from file, they are instead
    filtered out by this class.

    Each individuals file is a pickle.  It is named using random characters and
    its filename is assigned to attribute individual.filename.

    The scoreboard file is a comma separated values file containing the filename
    and fitnesses of every individual in the population.  It is kept in sorted
    order.  It is named "scoreboard.csv".  This class maintains attribute
    "scoreboard" for convenience.

    This class is a subclass of list and it maintains in itself the list of the
    most fit individuals in the population.  The maximum number of individuals
    to load is controlled by attribute population_size.

    Attribute population_size is the maximum number of individuals which this
              population will contain at one time.
    """
    def __init__(self, population_name, population_size):
        """
        Argument population_name ...
        Argument population_size is the number of Individuals to use.  If there
                 are more individuals than this on file then only the most fit
                 of them are used.  
        """
        super().__init__()
        self.population_name = population_name
        self.population_size = population_size
        program, program_ext = os.path.splitext(sys.argv[0])
        self.path            = os.path.join(program + '_data', self.population_name)
        # Load the scoreboard file.
        self.scoreboard_filename = os.path.join(self.path, 'scoreboard.csv')
        self.scoreboard = []
        try:
            with open(self.scoreboard_filename) as sbf:
                for entry in csv.DictReader(sbf):
                    entry['fitness'] = float(entry['fitness'])
                    self.scoreboard.append(entry)
        except FileNotFoundError:
            pass
        # Load the top of the scoreboard.
        for entry in self.scoreboard:
            if len(self) >= self.population_size:
                break
            file_path = os.path.join(self.path, entry['name'])
            try:
                with open(file_path, 'rb') as indv_file:
                    indv  = pickle.load(indv_file)
                self.append( indv )
            except FileNotFoundError:
                pass

    def generate_filename(self):
        length = 5
        taken_names = set(entry['name'] for entry in self.scoreboard)
        for attempt in range(10):
            random_tag = ''.join(chr(ord('A') + random.randrange(26)) for _ in range(length))
            name = random_tag + '.indv'
            if name not in taken_names:
                return name
        raise Exception("Could not generate unique file name.")

    def save(self, indv):
        """
        Adds an individual to this population.

        Argument indv is an instance of Individual.  This does NOT check if
                 the given individual has already been saved, don't save twice.
        """
        # Save individual to file
        if not hasattr(indv, 'filename'):
            indv.filename = self.generate_filename()
        file_path         = os.path.join(self.path, indv.filename)
        os.makedirs(self.path, exist_ok=True)
        with open(file_path, 'wb') as indv_file:
            pickle.dump(indv, indv_file)
        # Update the scoreboard.
        self.scoreboard.append({'name': indv.filename, 'fitness': indv.fitness})
        self.scoreboard.sort(reverse=True, key=lambda entry: entry['fitness'])
        with open(self.scoreboard_filename, 'w') as sbf:
            writer = csv.DictWriter(sbf, fieldnames=['name', 'fitness'])
            writer.writeheader()
            for entry in self.scoreboard:
                writer.writerow(entry)
        # Add to internal population
        self.append(indv)
        self.sort(reverse=True, key=lambda indv: indv.fitness)
        while len(self) > self.population_size:
            self.pop()

    def _average(self):
        """
        Returns an individual whos parameters are set to the statistical mean of
        the population's parameters.
        """
        1/0 # unimplemented
        indv = type(self[0])
        pop_stats = type(self[0]).population_statistics(self)


def evolutionary_algorithm(
    experiment_class,
    population,
    mutation_probability            = 0.20,
    mutation_percent                = 0.20,
    num_processes                   = None,
    profile                         = False,):
    """
    Argument experiment_class ... initializer is given no arguments.
    Argument population ...

    Argument mutation_probability ...
    Argument mutation_percent ...

    Argument num_processes ... defaults to the number of CPU cores present.
    Argument profile ...
    """
    if num_processes is None:
        num_processes = os.cpu_count()
    worker_pool       = mp.Pool(num_processes, maxtasksperchild=1)
    profile_stats     = []
    evals_done        = 0

    # The following two functions call each other to keep the process pool full
    # at all times.
    def start_worker_subproc():
        # Start with a blank individual, default parameters.
        indv = experiment_class()
        # Crossover and mutate.
        if len(population):
            indv.crossover(random.sample(population, min(2, len(population))))
        if random.random() < mutation_probability:
            indv.mutate(mutation_percent)
        # Start evaluating the individual.  The process pool will call the
        # callback (end_worker_subproc) with the results when it's done.
        subproc_arguments = (indv, profile)
        worker_pool.apply_async(_ea_subproc_main, subproc_arguments,
                                callback = end_worker_subproc)

    def end_worker_subproc(results):
        # Unpack the results of the evaluation.
        if profile:
            indv, fitness, pr_file = results
            profile_stats.append(pr_file)
        else:
            indv, fitness = results
        # Only save it to file if it did not crash.
        if fitness is not None:
            indv.apply_evaluate_return(fitness)
            population.save(indv)
        # Queue the next job.
        start_worker_subproc()
        # Book keeping
        nonlocal evals_done
        evals_done += 1

    try:
        # Kick off the evolutionary algorithm.
        for task in range(num_processes):
            start_worker_subproc()
            # Stagger the start-ups, which take lots of temporary memory.
            for second in range(60):
                time.sleep(1)   # Main thread remains responsive to subprocess communications.

        # Run until interrupted.
        prev_print = 0
        while True:
            if prev_print != evals_done:
                if evals_done % population.population_size == 0:
                    mean_fitness = sum(indv.fitness for indv in population) / len(population)
                    print("Mean fitness %g"%(mean_fitness))
                    prev_print = evals_done
            for second in range(30):
                time.sleep(1)   # Main thread remains responsive to subprocess communications.
    finally:
        if profile:
            # Assemble each evaluation's profile into one big profile and print it.
            stats = pstats.Stats(*profile_stats)
            stats.sort_stats('time')
            stats.print_stats()
            for tempfile in profile_stats:
                os.remove(tempfile)     # Clean up.

def _ea_subproc_main(individual, profile):
    """
    This function is executed in a subprocess.

    If individual.evaluate() raises a ValueError then its fitness is replaced
    with None, caller should discard this individual.

    Argument profile ... is boolean, causes this to return the path to a 
             stats_file which is a temporary file containing the binary
             pstats.Stats() output of the profiler.

    Returns either (individual, fitness)
            or     (individual, fitness, profile_stats_filename)
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    try:
        indv_inst_copy = copy.deepcopy(individual)
        fitness = indv_inst_copy.evaluate()
    except Exception:
        # This exception handler works because the process pool ends its
        # processes after every job and makes a new clean process for the next
        # job.
        print(str(individual))
        traceback.print_exc()
        fitness = None
    if profile:
        pr.disable()
        stats_file = tempfile.NamedTemporaryFile(delete=False)
        pr.dump_stats(stats_file.name)
        return individual, fitness, stats_file.name
    else:
        return individual, fitness


def memory_fitness(threshold=2e9, maximum=3e9):
    """
    Returns a penalty for using too much memory.  Add this to your fitness
    function.  This measures the current processes maximum resident set (maxrss)
    which is the all time peak memory usage for the calling process.

    Argument threshold is where the this penalty begins.
    Argument maximum is where this penalty becomes an error (ValueError).

    Returns in the range [0, 1] where 0 is no penalty and 1 is the maximum
            memory usage.  Linear ramp from threshold to maximum.
    """
    import resource
    rsc = resource.getrusage(resource.RUSAGE_SELF)
    size = rsc.ru_maxrss * 1024
    fit  = (size - threshold) / (maximum - threshold)
    if fit > 1:
        raise ValueError("Individual exceded memory limit (size %d bytes, maximum %d)."%(size, maximum))
    return max(0, fit)


class speed_fitness:
    """
    Inside of evaluate_function, use as such:
    1.  timer = speed_fitness(45 minutes, 90 minutes)
    2.  Do_work_which_may_take_a_long_time( instance_of_individual )
    3.  return { 'speed_fitness': timer.done() }
    Alternatively:
    3.  return { 'speed_fitness': timer.elapsed_time }
    4.  IndividualClass.fitness_names_and_weights['speed_fitness'] = 0
    """
    def __init__(self, threshold, maximum):
        """
        Arguments threshold, maximum ... units are minutes.
                  Raises a ValueError if the maximum time limit is exceded.
        """
        assert(signal.alarm(0) == 0) # Check that no alarm was already set.
        self.threshold   = threshold
        self.maximum     = maximum
        self.start_time  = time.time()
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(int(round(maximum * 60)))

    def handler(self, signum, frame):
        raise ValueError("Individual exceded time limit (%d minutes)."%self.maximum)

    def done(self):
        """
        Returned value is in range [0, 1], where 0 indicates that the speed is 
        sub-threshold and 1 is the maximum allowable time.
        """
        signal.alarm(0)          # Disable the alarm
        return max(0, (self.elapsed_time - self.threshold) / (self.maximum - self.threshold))

    @property
    def elapsed_time(self):
        """Minutes since this objects instantiation."""
        return (time.time() - self.start_time) / 60

