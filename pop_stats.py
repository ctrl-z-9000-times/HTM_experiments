#!/usr/bin/python3
# Written by David McDougall, 2017

import genetics
import argparse
import sys
import os, os.path

parser = argparse.ArgumentParser()
parser.add_argument('program_name', type=str)
parser.add_argument('population_name', type=str)
parser.add_argument('population_size', type=int)
parser.add_argument('--clone', type=str, default=None)
parser.add_argument('--clear_fitness', action='store_true')
parser.add_argument('--individuals',   action='store_true')
parser.add_argument('--mean',   action='store_true')
parser.add_argument('--max_children', type=int,
    default=genetics.ExperimentMain.ArgumentParser().get_default('max_children'))
args = parser.parse_args()

module = os.path.splitext(os.path.basename(args.program_name))[0]
exec('from ' + module + ' import *')
sys.argv[0] = args.program_name
pop = genetics.Population(args.population_name, args.population_size, args.max_children)

if not len(pop):
    print("Population does not exist.")
    exit(0)

if args.individuals:
    for index, indv in reversed(tuple(enumerate(pop))):
        print("Rank %d"%(index+1))
        print(indv)
        print()

if args.mean:
    mean = type(pop[0]).average(pop)
    print("Average of population")
    print(mean)
    print()

pop[0].pprint_population_statistics(pop)

if args.clone is not None:
    clone_pop = genetics.Population(args.clone, args.population_size)
    # Move the individuals across to the new population.
    for indv in pop:
        if args.clear_fitness:
            for attr in indv.fitness_names_and_weights.keys():
                setattr(indv, attr, None)
        clone_pop.save(indv)
    print("Cloned top %d from %s to %s"%(clone_pop.population_size, pop.path, clone_pop.path))
    # Explain what just happened.
    if args.clear_fitness:
        print("Set all clones fitnesses to -infinity, they will only be used as")
        print("seed genetic material, they will not compete with new individuals.")
    else:
        print("Did not clear the clones fitnesses, they will continue to compete")
        print("in the new population.")
elif args.clear_fitness:
    print("Will not clear fitness unless also cloning population.")
