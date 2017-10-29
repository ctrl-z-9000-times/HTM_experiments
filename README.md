# HTM_experiments

Welcome, these are my experiments with Hierarchial Temporal Memory (HTM).  HTM
is a computational model of neurons in the mammalian cerebral cortex.  For more
information about the theory, see: https://numenta.org

Written by David McDougall, 2017.

## Installation Notes:
TODO
- Install datasets repository, place link to it in this directory...
- `./setup.py build_ext --inplace`

## MNIST Experiment Notes:

The MNIST dataset contains small images of the numbers 0-9.  The dataset is
split into 60,000 training and 10,000 testing images.  The goal of this
experiment is to make a program which can look at the labeled training data and
then correctly identify the unlabeled testing data.

MNIST is used as a fast check for my spatial pooler and synapse manager code.
The command `$ ./mnist_sp.py --default_parameters` should yield a score of at
least 0.93 out of a best possible score of 1.00.

## Eye Experiment Notes:
  TODO

## Image Dataset Notes:
The datasets repository is for creating and manipulating datasets of labeled
images.  It contains a python3 module and a GUI tool, named label_tool.py.  The
program label_tool.py is used to paint areas of the image with labels for the AI
to identify.  For help see label_tool.py's in-program help message.

## Genetic Algorithm Notes:

All experiments use a simple evolutionary algorithm because tuning these things
manually is time consuming and usually results in sub-optimal performance.  The
code is located in module genetics.py.  The evolutionary algorithm creates
genetic material for new individuals by mixing up the genetic material of two
existing members of the population and sometimes applying a mutation too.  A
pool of processes evaluates new individuals until you interrupt it (with
CTRL-C).  You can view a population and statistics about it with the program
pop_stats.py.

## Channel Encoder Notes:

These encoders transform inputs such as images into sparse distributed
representations.  Two requirements for SDRs are that every bit of information
represents a range of possible values, and that for every input a certain
fraction of bits activate in the SDR.  The effects of these requirements are
that each output bit is inaccurate and redundant.  This implementation makes
every output bit receptive to a unique range of inputs.  These ranges are
uniformly distributed through the inputs space and the widths are the size of
the input space multiplied by the target sparsity.  This meets both requirements
for being an SDR and has more representational power than if many of the bits
represented the same ranges. This design makes all of those redundancies add
useful information.

## Eye Sensor Notes:

The eye sensor is like a human eye; it is foveated and can move about an image.
The eye sensors retina has a receptor dense central fovea and a receptor sparse
perifieral vision.  It uses the channel encoders to convert hue, saturation,
value and edge information into an SDR which has topology defined.  Its location
is controlled using an SDR and it outputs its positions and movements encoded as
an SDR.

## Spatial Pooler Notes:

This implementation is based on but differs from the one described by Numenta's
Spatial Pooler white paper, (Cui, Ahmad, Hawkins, 2017, "The HTM Spatial Pooler
\- a neocortical...", https://doi.org/10.1101/085035) in two main ways, the
boosting function and the local inhibition mechanism.

### Logarithmic Boosting Function:

This uses a logarithmic boosting function.  Its input is the activation
frequency which is in the range [0, 1] and its output is a boosting factor
to multiply each columns excitement by.  It's equation is: `boost-factor = log(
activation-frequency ) / log( target-frequency )`.  Some things to note:
1. The boost factor asymptotically approaches infinity as the activation
frequency approaches zero.
2. The boost factor equals zero when the activation frequency is one.
3. The boost factor for columns which are at the target activation frequency is
one.
4. This mechanism has a single parameter: boosting_alpha which controls the
exponential moving average which tracks the activation frequency.

### Fast Local Inhibition:

This activates the most excited columns globally, after normalizing all
columns by their local area mean and standard deviation.  The local area is
a gaussian window and the standard deviation of it is proportional to the
deviation which is used to make the receptive field of each column.
Columns inhibit each other in proportion to the number of inputs which they
share.  In pseudo code:

```
mean_normalized = excitement - gaussian_blur( excitement, radius )
standard_deviation = sqrt( gaussian_blur( mean_normalized ^ 2, radius ))
normalized = mean_normalized / standard_deviation
activate = top_k( normalized, sparsity * number_of_columns )
```

The reason I defined a neurons local neighborhood using normal distributions is
because they are both linearly seperable and have rotational symetry.  Linearly
seperable is a mathematical property which in short, makes gaussian filters very
fast to compute.  Normal distributions are unique in having both of these
properties.

## Reinforcement Learning Notes:

Many thanks to:  
Sungur, 2017.  "HIERARCHICAL TEMPORAL MEMORY BASED AUTONOMOUS AGENT FOR
PARTIALLY OBSERVABLE VIDEO GAME ENVIRONMENTS"

TODO: explain all of the ways in which my system is different from Sungur's
system.



## Cython Notes:

The htm module is written in Cython, a superset of python3 which is compiled
into C code and then to binary python3 extension module.  Cython adds C types to
python3 code and automatically inserts the interfaces needed to go between C
types and python3 objects.  Code using these C types bypasses the python3 type
system and runs at the speed of native C code.  For more about Cython see:
http://cython.org/.  To build the htm Cython module use: `$ ./setup.py
build_ext --inplace`

## Synapse Manager Notes:
The synapse manager class implements an NMDA receptor, which functions as a
coincidence detector.  NMDA receptors detects when an input activation implies
that an output is likely to activate as well.

Profiling shows that notifying each output about which inputs also activated
is the one of the more time consuming steps.  To facilitate this step, this
class keeps an extra index table containing which outputs are connected to by
each input. Using this optimization, synapses are only visited if their
presynaptic input is active and if the synapse is in a connected state.
