#!/usr/bin/python3
# Written by David McDougall, 2017

import htm
from mnist_sp import *
from tp_numbers_test import *
from eye_experiment import *
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_filename', type=str)
parser.add_argument('--remove_fitness', action='store_true')
args = parser.parse_args()

pop = pickle.load(open(args.checkpoint_filename, 'rb'))

if args.remove_fitness:
    backup = args.checkpoint_filename + '.fitness'
    print("BACKING UP POPULATION TO %s"%backup)
    with open(args.checkpoint_filename, 'rb') as inp:
        with open(backup, 'wb') as out:
            out.write(inp.read())
    print("REMOVING ATTRIBUTE FITNESS FROM POPULATION %s"%args.checkpoint_filename)
    for indv in pop:
        indv.clear_fitness()
    pickle.dump(pop, open(args.checkpoint_filename, 'wb'))
    exit(0)

for indv in pop:
    print(str(indv))
    print()
print()
type(pop[0]).pprint_population_statistics(pop)

