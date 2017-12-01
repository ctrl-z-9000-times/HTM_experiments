"""
Evolutionary algorithms and supporting tools.

Written by David McDougall, 2017
"""

import numpy as np
import random
import multiprocessing as mp
import traceback
import cProfile, pstats, tempfile
import sys
import time
import resource
import signal
import os, os.path
import pickle
import copy
import csv
import argparse

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

    def _choose_parameter(self):
        """This assigns all parameters uniform probability of mutation,
        except None which is never mutated."""
        def attr_probability(attr):
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
        probabilities = [attr_probability(attr) for attr in self.parameters]
        probabilities = np.array(probabilities) / np.sum(probabilities)
        param_idx     = np.random.choice(len(self.parameters), p=probabilities)
        parameter     = self.parameters[param_idx]
        return parameter

    def mutate_standard(self, percent, population):
        """
        Randomly change some parameters.  The change in value uses the populations
        standard deviation.
        """
        def mutate_value(value, pop_values):
            if random.random() > percent:
                return value

            pop_values = [v for v in pop_values if v is not None]
            if len(np.unique(pop_values)) < 3:
                # Use alternative method when diversity is very low.
                return value * 1.5 ** (random.random()*2-1)
            else:
                std = np.std(pop_values)
                return float(random.gauss(value, std))

        for param in self.parameters:
            value      = getattr(self, param)
            pop_values = [getattr(indiv, param) for indiv in population]

            if value is None:
                continue # cant mutate.

            elif isinstance(value, Parameters):
                value.mutate_standard(percent, pop_values)

            elif isinstance(value, tuple):
                new_tup = []
                for index, value_indexed in enumerate(value):
                    pop_values_indexed = [v[index] for v in pop_values]
                    new_value = mutate_value(value_indexed, pop_values_indexed)
                    new_tup.append(new_value)
                setattr(self, param, tuple(new_tup))

            else:   # Mutate a floating point or boolean number.
                setattr(self, param, mutate_value(value, pop_values))

    def mutate(self, percent):
        """
        Randomly change a single parameter by at most the given percent.
        This uses an exponential mutation rate: (1 + percent) ^ uniform(-1, 1)
        """
        param = self._choose_parameter()
        value = getattr(self, param)
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
        """Returns the number of non-None parameters which this instance contains."""
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
        s = self.__module__ + '.' + type(self).__name__ + "(\n"
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
    def average(cls, population, instance=None):
        """
        Returns an instance of this parameter class whose calues are the
        statistical mean of the populations values.
        """
        if not population:
            raise ValueError("Population is empty.")
        if instance is None:
            instance = cls()
        for param in cls.parameters:
            default    = getattr(instance, param)
            pop_values = [getattr(indiv, param) for indiv in population]
            if isinstance(default, Parameters):
                pop_mean = default.average(pop_values, default)
            else:
                if all(v is None for v in pop_values):
                    pop_mean = None
                elif isinstance(default, bool):
                    pop_mean = bool(round(np.mean(pop_values)))
                elif isinstance(default, tuple):
                    pop_mean = tuple(np.mean(pop_values, axis=0))
                else:   # Floating point case.
                    pop_mean = np.mean(pop_values)
            setattr(instance, param, pop_mean)
        return instance

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
            # if len(data_types) != 1:
            #     raise TypeError("Not one data type for attribute '%s', found %s"%(attr, data_types))
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
        # Push fitness and resources to either the top of the bottom of the
        # table and then sort alphabetically.
        sort_order = lambda entry: (
            entry[0].count('fitness.fitness'), # Awalys push this entry to bottom of table.
            entry[0].count('fitness.'),
            entry[0].count('resources.'),
            entry[0])
        entries.sort(key = sort_order)
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
    possible for the Population class and evolutionary algorithm to make new
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

    Attribute debug ...
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

    def evaluate(self, debug):
        """
        Subclass should override this method to measure the fitness of this set
        of parameters.

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

    def _timeout_handler(self, signum, frame):
        raise ValueError("Individual exceded time limit (%d minutes)."%self._timeout)

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
        Combines Individual's fitness and resource usage statistics with the
        parent classes parameter statistics.
        """
        table = super().population_statistics(population)
        fitness_stats = cls.population_fitness(population)
        fitness_stats = {'fitness.'+attr:stat for attr, stat in fitness_stats.items()}
        table.update(fitness_stats)

        # Resource usage statistics.
        population_resources = [getattr(indv, 'resources', None) for indv in population]
        population_resources = [r for r in population_resources if r is not None]
        if population_resources:
            resource_keys = set()
            resource_keys.update(*[d.keys() for d in population_resources])
            resource_stats = {}
            for key in resource_keys:
                data = [dic.get(key, None) for dic in population_resources]
                if any(datum is None for datum in data):
                    resource_stats[key] = None
                else:
                    resource_stats[key] = (
                        np.min(data),
                        np.mean(data),
                        np.std(data),
                        np.max(data),)
            resource_stats = {'resources.'+key:stat for key, stat in resource_stats.items()}
            table.update(resource_stats)

        return table

    # TODO: This method should show the resource usage if available.
    def __str__(self):
        """Combines the parents parameter table with a fitness table."""
        param_table = super().__str__()
        entries = [(attr, getattr(self, attr, None)) for attr in self.fitness_names_and_weights.keys()]
        entries.sort(key = lambda attr_value: attr_value[0])
        entries.append(("Weighted Fitness", self.fitness))
        # I'm confused by the following line.  I dont think it does anything.
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

def _Individual_call_evaluate(self, debug, profile, timeout=None, memory_limit=None):
    """
    The multiprocessing library needs this method to be in global scope.  It is
    also why the method needs to return self.

    Argument debug is passed to evaluate.

    Argument profile ... is boolean, causes this to return the path to a 
             stats_file which is a temporary file containing the binary
             pstats.Stats() output of the profiler.

    Optional Argument timeout is in minutes, will interrupt the evaluate if enabled.
    Optional Argument memory_limit is in bytes, sets OS limit on resident memory usage.

    Returns either self or (self, profile_stats_filename)

    Attibute results is the dictionary which evaluate returns.  If evaluate
                     raises an exception this is set to None.
    Attibute resources is a dictionary containing keys "bytes" and "minutes".
    """
    start_time = time.time()
    # Setup Profiling
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    # Setup memory resource limits
    if memory_limit is not None:
        1/0 # unimplemented.
    # Setup timeout alarm
    if timeout is not None:
        self._timeout = timeout
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(int(round(timeout * 60)))
    # Evaluate fitness
    inst_copy = copy.deepcopy(self)
    try:
        self.results = inst_copy.evaluate(debug)
    except Exception:
        print(str(self))
        traceback.print_exc()
        self.results = None
    # Disable the timeout alarm
    if timeout is not None:
        signal.alarm(0)
    # Measure resource usage
    rsc = resource.getrusage(resource.RUSAGE_SELF)
    self.resources = {
        'bytes':   rsc.ru_maxrss * 1024,
        'minutes': (time.time() - start_time) / 60
    }
    # Collect profiler data and return
    if profile:
        pr.disable()
        stats_file = tempfile.NamedTemporaryFile(delete=False)
        pr.dump_stats(stats_file.name)
        return self, stats_file.name
    else:
        return self
Individual.call_evaluate = _Individual_call_evaluate


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


def evolutionary_algorithm(
    experiment_class,
    population,
    mutation_percent                = 0.10,
    num_processes                   = None,
    profile                         = False,):
    """
    Argument experiment_class ... initializer is given no arguments.
    Argument population ...

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
        indv.mutate_standard(mutation_percent, population)
        # Start evaluating the individual.  The process pool will call the
        # callback (end_worker_subproc) with the results when it's done.
        debug = False
        subproc_arguments = (indv, debug, profile,)
        worker_pool.apply_async(_Individual_call_evaluate, subproc_arguments,
                                callback = end_worker_subproc)

    def end_worker_subproc(returned_values):
        # Unpack the results of the evaluation.
        if profile:
            indv, pr_file = returned_values
            profile_stats.append(pr_file)
        else:
            indv = returned_values
        # Only save it to file if it did not crash.
        if indv.results is not None:
            indv.apply_evaluate_return(indv.results)
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
            gen_size = population.population_size
            if evals_done // gen_size > prev_print // gen_size:
                mean_fitness = sum(indv.fitness for indv in population) / len(population)
                print("Mean fitness %g"%(mean_fitness))
                prev_print = evals_done
            for second in range(30):
                time.sleep(1)   # Main thread remains responsive to subprocess communications.
    except KeyboardInterrupt:
        print("^KeyboardInterrupt")
        print()
    finally:
        if profile:
            # Assemble each evaluation's profile into one big profile and print it.
            stats = pstats.Stats(*profile_stats)
            stats.sort_stats('time')
            stats.print_stats()
            for tempfile in profile_stats:
                os.remove(tempfile)     # Clean up.


class ExperimentMain:
    """
    This provides a main function for using this module from the command line.

    Example:
    >>> if __name__ == '__main__':
    >>>     arg_parser = genetics.ExperimentMain.ArgumentParser()
    >>>     arg_parser.add_argument('--my_settings', ...)
    >>>     args = arg_parser.parse_args()
    >>>     ExperimentClass.settings = args.my_settings
    >>> 
    >>>     # The following line runs the main function.
    >>>     genetics.ExperimentMain(ExperimentClass, arg_parser)
    """
    @classmethod
    def ArgumentParser(cls, description=None, epilog=None):
        """
        Argument parser for genetic experiments. 

        Optional Arguments description and epilog are printed in the help message
            before and after the argument listing.  
        """
        if description is None:
            description = ''    # TODO
        if epilog is None:
            epilog = ''         # TODO
        arg_parser = argparse.ArgumentParser(
            description = description,
            epilog = epilog)
        arg_parser.add_argument('--default_parameters', action='store_true',
            help='Evaluate the default parameters. Sets Debug flag.')
        arg_parser.add_argument('--best_parameters',    action='store_true',
            help='Evaluate the best parameters in the population. Sets Debug flag.')
        arg_parser.add_argument('--mean_parameters',    action='store_true',
            help='Evaluate the average parameters of the population. Sets Debug flag.')
        arg_parser.add_argument('--file', type=str,   default='checkpoint',
            help='What name to save the results by.')
        arg_parser.add_argument('-p', '--population', type=int, default=100)
        arg_parser.add_argument('-n', '--processes',  type=int, default=4,
            help="Number of worker processes to use.")
        arg_parser.add_argument('-m', '--mutate',     type=float, default=0.05,
            help='What fraction of parameters are mutated at conception.')
        arg_parser.add_argument('--profile',          action='store_true')
        return arg_parser

    def __init__(self, experiment_class, arg_parser=None):
        """
        Argument experiment_class is an instance of Individual.
        Optional Argument arg_parser ... always  use ExperimentMain.ArgumentParser
            to create this argument, then add/modify arguments as needed.
        """
        if arg_parser is None:
            arg_parser = self.ArgumentParser()
        args = arg_parser.parse_args()

        if args.default_parameters or args.best_parameters or args.mean_parameters:
            if args.default_parameters:
                indv = experiment_class()
                print("Default Parameters")
            elif args.best_parameters:
                indv = Population(args.file, 1)[0]
                print("Best of population")
            elif args.mean_parameters:
                population = Population(args.file, args.population)
                indv = experiment_class.average(population)
                print("Average of population")

            print(indv)
            print()
            result = indv.call_evaluate(debug=True, profile=args.profile)
            if args.profile:
                indv, prof = result
                stats = pstats.Stats(prof)
                stats.sort_stats('time')
                stats.print_stats()
                os.remove(prof)     # Clean up
            print("Evaluate returned", indv.results)
            print("Resources used", indv.resources)
        else:
            population = Population(args.file, args.population)
            evolutionary_algorithm(
                experiment_class,
                population,
                mutation_percent                = args.mutate,
                num_processes                   = args.processes,
                profile                         = args.profile,)


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

