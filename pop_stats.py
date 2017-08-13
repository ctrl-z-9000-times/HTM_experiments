#!/usr/bin/python3
# Written by David McDougall, 2017

import htm
from mnist_sp import *
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_filename', type=str)
args = parser.parse_args()

pop = pickle.load(open(args.checkpoint_filename, 'rb'))

if False:
    print("REMOVING ATTRIBUTE FITNESS")
    for indv in pop:
        indv.fitness = None
    pickle.dump(pop, open(args.checkpoint_filename, 'wb'))

type(pop[0]).pprint_population_statistics(pop)
print()
for indv in pop:
    print(str(indv))
    if getattr(indv, 'fitness', None) is not None:
        print("Fitness", indv.fitness)
    else:
        print("Fitness not yet evaluated...")
    print()

