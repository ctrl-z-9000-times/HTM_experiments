# Written by David McDougall, 2017

# TODO: Module level docstring

import numpy as np
import random
import multiprocessing as mp
import cProfile, pstats, tempfile
import traceback
import sys
import itertools
import collections
import time
import signal
import re
import os, os.path
import pickle
import copy

class Parameters:
    """
    Abstract parent class for genetic material.  Child classes represent
    parameters for a specific type of object.  The purpose of this class is to
    facilitate a parameter search, for example a genetic algorithm or swarming.

    Class attribute parameters is the list of attributes which make up the 
    mutable parameters, all other attributes are ignored by this class and its
    methods.  The type of these attributes must be one of:
        1) Floating point number
        2) Tuple of floating point numbers
        3) Instance of Parameters or a child class.
    """
    parameters = []

    def __init__(self, **kw_args):
        """
        Child classes override this as needed.  This writes all keyword
        arguments into the new instance as its initial values.
        """
        p_set = set(self.parameters)
        for attr, value in kw_args.items():
            if attr not in p_set:
                raise TypeError("'%s' is not a valid keyword argument to this class"%attr)
            setattr(self, attr, value)
        missing_parameters = [attr for attr in self.parameters if not hasattr(self, attr)]
        if missing_parameters:
            raise TypeError("GA_Parameter.__init__() missing parameters: %s"%(', '.join(missing_parameters)))

    def mutate(self, percent=0.25):
        """
        Randomly change a single parameter by at most the given percent.
        Default maximum mutation is 25%
        """
        # Pick a parameter to mutate and get its value.
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
                return 1
        probabilities   = [attr_probability(attr) for attr in self.parameters]
        probabilities   = np.array(probabilities) / np.sum(probabilities)
        param_idx       = np.random.choice(len(self.parameters), p=probabilities)
        param           = self.parameters[param_idx]
        value           = getattr(self, param)
        if isinstance(value, Parameters):
            value.mutate(percent)
        else:
            modifier = (1 - percent) + 2 * percent * random.random()
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
        """
        for attr in self.parameters:
            values = [getattr(obj, attr) for obj in parents]
            if isinstance(values[0], Parameters):
                getattr(self, attr).crossover(parents)
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
        indent    = 2
        header    = [type(self).__name__]
        table     = []
        max_name  = max(len(nm) for nm in self.parameters)
        for attr in sorted(self.parameters):
            pad   = max_name - len(attr) + 2
            value = str(getattr(self, attr, None))
            if '\n' in value:
                value = value.replace('\n', '\n'+' '*indent)
                value = value.split('\n', maxsplit=1)[1]      # Hacks
                table.append(' '*indent + attr +'\n'+ value)
            else:
                table.append(' '*indent + attr +' '+ '.'*pad +' '+ value)
        table.sort(key = lambda ln: '\n' in ln)
        table = '\n'.join(table)

        # Make all of the columns line up.
        align_to = max(len(entry) for entry in re.findall(r"^.*\.\.\s", table, re.MULTILINE))
        aligned_table = []
        for line in table.split('\n'):
            match = re.match(r"(^.*\.\.\s)", line, re.MULTILINE)
            if match is not None:
                extend = align_to - len(match.group())
                line = line.replace('.. ', '.'*extend + '.. ')
            aligned_table.append(line)
        return '\n'.join(header + aligned_table)

    @classmethod
    def population_statistics(cls, population):
        """
        Finds the population wide minimum, mean, standard-deviation, and
        maximumn of all parameters.  The special attribute 'fitness' is included
        if it is present and not None.  

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
            data       = [getattr(indv, attr) for indv in population]
            data_types = set(type(v) for v in data)
            if int in data_types and float in data_types:
                data_types.remove(int)  # Integers are not part of the spec but are OK.
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
        lines      = ["Population Statistics for %d %s"%(len(population), cls.__name__)]
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
    This class provides a fitness function and several supporting methods. The
    fitness function is a weighted sum; its variables are the attributes of
    this class which are named in fitness_names_and_weights.keys(),
    fitness_names_and_weights[ attrbute-name ] is the weight which is used.
    """
    fitness_names_and_weights = {}

    @property
    def fitness(self):
        """
        Returns the weighted sum of fitnesses, ignoring any attributes set to 
                None.
        """
        memo = getattr(self, '_fitness', None)
        if memo is not None:
            return memo
        can_memo = True
        # Calculate the current fitness.
        fitness = None
        for attr, weight in self.fitness_names_and_weights.items():
            value = getattr(self, attr, None)
            if value is not None:
                if fitness is None:
                    fitness = 0
                fitness += value * weight
            else:
                can_memo = False
        # Only save it the fitness if more data won't be arriving.
        if can_memo:
            self._fitness = fitness
        return fitness

    def clear_fitness(self):
        for attr in self.fitness_names_and_weights.keys():
            setattr(self, attr, None)
        self._fitness = None

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
        indent = 2
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
            lines = [' '*indent + 'fitness']
            for attr, value in entries:
                pad = max_name - len(attr) + 2
                lines.append('  '*indent + attr +' '+ '.'*pad +' '+ str(value))
            fitness_table = '\n'.join(lines)
        return param_table + '\n' + fitness_table


def genetic_algorithm(parameter_class, evaluate_function,
    population_size                 = 50,
    num_epochs                      = 1,
    seed                            = False,
    seed_mutations_per_parameter    = 2,
    seed_mutation_percent           = 0.025,
    mutation_probability            = 0.10,
    mutation_percent                = 0.25,
    filename                        = 'checkpoint',
    num_processes                   = 7,
    profile                         = False,):
    """
    Argument parameter_class ... is called with no arguments and must return an
             instance of Individual.

    Argument evaluate_function is called with an instance of the parameter_class
             and must return a dict whos names are fitness_names_and_weights.keys()
             and their values or None placeholders.  
             If evaluate_function raises a ValueError then this function will
             assume the indivual's parameters are at fault and discards them
             from the population by setting their fitness to negative infinity.

    Note: All extra arguments to parameter_class and evaluate_function must be
          applied before they are given to this function.  See functools.partial.

    Argument population_size ...
    Argument num_epochs ...
    Argument mutation_probability ...
    Argument mutation_percent ...

    Argument seed ...
    Argument seed_mutations_per_parameter ...
    Argument seed_mutation_percent is the amount to mutate the seeds by.

    Argument filename is passed to checkpoint() to save the results.
    Argument num_processes ...
    Argument profile ...
    """
    assert(population_size >= 2)    # Can't/Won't reproduce asexually.
    population    = []
    generation    = 0
    profile_stats = []
    if seed:
        # Make the initial population
        print("SEEDING WITH %d"%population_size)
        for _ in range(population_size):
            indv = parameter_class()
            for mut in range(int(round(seed_mutations_per_parameter * len(indv)))):
                indv.mutate(seed_mutation_percent)
            population.append(indv)
    # Run the Genetic Algorithm
    for epoch in range(num_epochs):
        population, generation = checkpoint(population, generation, filename)
        print("CHECKPOINT %d, POP SZ %d, %s"%(generation, len(population), filename))
        # Generate new parameters and new combinations of parameters.
        for _ in range(population_size):
            chd = parameter_class()
            chd.crossover(random.sample(population, 2))
            if random.random() < mutation_probability:
                chd.mutate(mutation_percent)
            population.append(chd)
        # Evaluate each individual
        eval_list = [indv for indv in population if indv.fitness is None]
        eval_args = [(indv, evaluate_function, profile) for indv in eval_list]
        with mp.Pool(num_processes) as procpool:
            results = procpool.starmap(_genetic_algorithm_evaluate, eval_args)
        # Deal with the results of the evaluations.
        if profile:
            fitness_list, profiles = zip(*results)
            profile_stats.extend(profiles)
        else:
            fitness_list = results
        for indv, fit_dict in zip(eval_list, fitness_list):
            # Put the fitnesses where they belong.
            for attr_name, fitness in fit_dict.items():
                setattr(indv, attr_name, fitness)
        # Cull the population down to size.
        population.sort(key=lambda indv: indv.fitness, reverse=True)
        population = population[:population_size]
        generation += 1
        print("Mean fitness %g"%(sum(indv.fitness for indv in population)/len(population)))

    if profile:
        # Assemble each evaluation's profile into one big profile and print it.
        stats = pstats.Stats(*profile_stats)
        stats.sort_stats('time')
        stats.print_stats()
        for tempfile in profile_stats:
            os.remove(tempfile)     # Clean up

    population, generation = checkpoint(population, generation, filename)
    print("SUMMARY FOR GENERATION %d, %s"%(generation, filename))
    parameter_class.pprint_population_statistics(population)
    print()
    for indv in population[:5]:
        print(str(indv))
        print()
    return population, generation

def _genetic_algorithm_evaluate(individual, evaluate_function, profile):
    """
    This function is executed in a subprocess.

    Argument profile ... is boolean, causes this to return tuple of (fitness,
             stats_file) where stats_file is a temporary file containing the
             binary pstats.Stats() output of the profiler.
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    try:
        fitness = evaluate_function(individual)
        # Check that evaluate_function returned a valid result.
        if fitness.keys() != individual.fitness_names_and_weights.keys():
            raise TypeError("Evaluate function returned dictionary which has missing/extra keys.")
    except ValueError:
        print(str(individual))
        traceback.print_exc()
        fitness = float('-inf')
    finally:
        signal.alarm(0)     # Cancels any timer which class speed_fitness set.
    if profile:
        pr.disable()
        stats_file = tempfile.NamedTemporaryFile(delete=False)
        pr.dump_stats(stats_file.name)
        return fitness, stats_file.name
    else:
        return fitness


def checkpoint(population, generation, filename='checkpoint'):
    """
    Saves/Loads the population from file.

    Folder is name of program + "_data/".
    Filename is "checkpoint.[GENERATION].pop"

    If the population is empty and there is a checkpoint on file then the
    latest checkpoint is returned as a pair of (population, generation).
    Otherwise this returns the given (population, generation).
    """
    program, ext  = os.path.splitext(sys.argv[0])
    path, program = os.path.split(program)
    folder        = os.path.join(path, program + '_data')

    def save(pop, gen, filename):
        filename_ext = filename + '.' + str(gen) + '.pop'
        full_filename = os.path.join(folder, filename_ext)
        try:
            os.makedirs(folder)
        except OSError:
            pass
        pickle.dump(pop, open(full_filename, 'wb'))

    def max_gen(filename):
        """Returns pair of (filename, generation)"""
        matches = []
        for fn in os.listdir(folder):
            fn_format = r'^' + filename + r'\.(\d+)\.pop$'
            m = re.match(fn_format, fn)
            if m:
                gen = int(m.groups()[0])
                fn  = os.path.join(folder, fn)
                matches.append((fn, gen))
        try:
            return max(matches, key=lambda p: p[1])
        except ValueError:
            return None, -1

    if len(population):
        save(population, generation, filename)
        return population, generation
    else:
        filename, latest_gen = max_gen(filename)
        if filename is not None and latest_gen >= 0:
            return pickle.load(open(filename, 'rb')), latest_gen
        else:
            return population, generation


def _total_size(o, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    If none of the handlers match, then this attempts to crawls obj.__dict__.
    """
    dict_handler = lambda d: itertools.chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    collections.deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)   # estimate size of object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        else:
            if hasattr(o, '__dict__'):
                attr = [val for name, val in o.__dict__.items() if not name.startswith('_')]
                s += sum(map(sizeof, attr))
        return s

    return sizeof(o)


def memory_fitness(instance, threshold=0.8e9, maximum=1.2e9):
    """
    Returns a penalty for using too much memory.  Add this to your fitness 
    function.

    Argument threshold is where the this penalty begins.
    Argument maximum is where this penalty becomes an error.

    Returns in the range [0, 1] where 0 is no penalty and 1 is the maximum
            memory usage.  Linear ramp from threshold to maximum.
    """
    size = _total_size(instance)
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
    """
    def __init__(self, threshold, maximum):
        """
        Argument threshold, maximum units are minutes.
                 Raises a ValueError if the maximum time limit is exceded.
        """
        assert(signal.alarm(0) == 0) # Check that no alarm was already set.
        self.threshold   = threshold
        self.maximum     = maximum
        self.start_time  = time.time()
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(int(round(maximum * 60)))

    def handler(self, signum, frame):
        raise ValueError("Individual exceded time limit.")

    def done(self):
        """
        Returned value is in range [0, 1], where 0 indicates that the speed is 
        sub-threshold and 1 is the maximum allowable time.
        """
        signal.alarm(0)          # Disable the alarm
        self.elapsed_time = (time.time() - self.start_time) / 60
        return max(0, (self.elapsed_time - self.threshold) / (self.maximum - self.threshold))

