#!/usr/bin/python3
"""
Written by David McDougall, 2017
All texts are from project Guttenburg.

The purpose of this experiment is to test the stability mechanism.

EXPERIMENT: Analyse state names.  Dataset is the 50 names in random order.  Then
take a very close look at when it resolves the difference between missouri and
missisippi, In theory: the predictive accuracy should increase after the "miss".
Also in theory: the stability should dip between objects, which means between
words and between sections of words, in this example miss\/ouri miss\/issipi.

EXPERIMENT: Introduce typos in the context of a sentance...   A verifiable
result of the stability mechanism is that it will become robust to misspellings
and typoes.  I can measure the resistance as word classification accuracy.
Already have code to generate typoes. Then turn off min_stability and check that
it loses its typo resistance.
** Apply typos to the example sentance which is displayed in detail, then get
** two graphs: one for original, another for typo.

"""

import numpy as np
import random
import itertools
import re
import os, os.path
import genetics
from matplotlib import pyplot as plt

import sdr              # Lowercase for module
from sdr import SDR     # Uppercase for class
import encoders
import unified
import classifiers

dictionary_file = '/usr/share/dict/american-english'

def read_dict():
    with open(dictionary_file) as f:
        return f.read().split()

def read_corpus(debug=False):
    """
    Converts to all lowercase.
    Removes excess whitespace.
    Converts all whitespace to space characters.
    """
    # Make a list of all available texts.
    data_path = 'datasets/ascii'
    books     = os.listdir(data_path)
    random.shuffle(books)   # Pick books at random.
    prev_char_is_space = False
    while True:
        try:
            selected_book = books.pop()
        except IndexError:
            raise StopIteration()
        if debug:
            print("Reading", selected_book)
        selected_book_path = os.path.join(data_path, selected_book)
        with open(selected_book_path) as file:
            # Read every letter of the book.
            while True:
                char = file.read(1)
                # Check for end of book, break to outer loop for next book.
                if not char:
                    break
                # Slam lower.
                char = char.lower()
                # Skip excess whitespace and convert all whitespace to literal spaces.
                if char.isspace() or char == '_':
                    if prev_char_is_space:
                        continue
                    char = ' '
                    prev_char_is_space = True
                else:
                    prev_char_is_space = False

                yield char

state_names = [
    'Alabama',
    'Alaska',
    'Arizona',
    'Arkansas',
    'California',
    'Colorado',
    'Connecticut',
    'Delaware',
    'Florida',
    'Georgia',
    'Hawaii',
    'Idaho',
    'Illinois',
    'Indiana',
    'Iowa',
    'Kansas',
    'Kentucky',
    'Louisiana',
    'Maine',
    'Maryland',
    'Massachusetts',
    'Michigan',
    'Minnesota',
    'Mississippi',
    'Missouri',
    'Montana',
    'Nebraska',
    'Nevada',
    'New Hampshire',
    'New Jersey',
    'New Mexico',
    'New York',
    'North Carolina',
    'North Dakota',
    'Ohio',
    'Oklahoma',
    'Oregon',
    'Pennsylvania',
    'Rhode Island',
    'South Carolina',
    'South Dakota',
    'Tennessee',
    'Texas',
    'Utah',
    'Vermont',
    'Virginia',
    'Washington',
    'West Virginia',
    'Wisconsin',
    'Wyoming',
]
def state_name_reader(debug=False):
    if debug:
        print("Reading states names dataset.")
    while True:
        state = random.choice(state_names)
        for char in state:
            yield char
        yield ' '


def mutate_word(word):
    """Introduce a random change into the word: delete, swap, repeat, and add
    stray character.  This may raise a ValueError.  """
    word = list(word)
    choice = random.randrange(4)
    if choice == 0:     # Delete a character
        word.pop(random.randrange(len(word)))
    elif choice == 1:   # Swap two characters
        index = random.randrange(0, len(word) - 1)
        word[index], word[index + 1] = word[index + 1], word[index]
    elif choice == 2:   # Repeat a character
        index = random.randrange(0, len(word))
        word.insert(index, word[index])
    elif choice == 3:   # Insert a stray character
        char = chr(random.randint(ord('a'), ord('z')))
        word.insert(random.randint(0, len(word)), char)
    return ''.join(word)


class ASCII_Experiment(genetics.Individual):
    parameters = genetics.Individual.parameters + [
        'enc_bits',
        'enc_sparsity',
        'l4',
        'l4_radii',
        'l23',
        'l23_radii',
        'sdrc',
    ]
    fitness_names_and_weights = {
        'L23_accuracy':     +1.0,
        'L23_anomaly':       0,
        'L23_end_accuracy':  0,
        'L23_stability':     0,
        'L4_accuracy':       0,
        'L4_anomaly':        0,
        'L4_end_accuracy':  +1.0,
        'memory':           -0.0005,
        'speed':            -0.005,
    }

    def __init__(self,):
        super().__init__()


        self.enc_bits = 3162
        self.enc_sparsity = 0.0418766768173
        self.l23 = unified.UnifiedParameters(
            AP                          = 42,
            boosting_alpha              = 0.00118565266738,
            cells                       = 5621,
            distal_add_synapses         = 15,
            distal_init_dist            = (0.3535624410183128, 0.20629559985411591),
            distal_initial_segment_size = 33,
            distal_learning_threshold   = 1,
            distal_mispredict_dec       = 0.000659232038862,
            distal_permanence_dec       = 0.00770685670518,
            distal_permanence_inc       = 0.0277663483461,
            distal_permanence_thresh    = 0.313938516638,
            distal_predicted_boost      = 4,
            distal_predictive_threshold = 8,
            distal_segments_per_cell    = 19,
            distal_synapses_per_segment = 32,
            min_stability               = 0.412610994787,
            predicted_proximal_boost    = 0.264201890295,
            proximal_active_thresh      = 4,
            proximal_add_synapses       = 11,
            proximal_dec                = 0.00075660737499,
            proximal_inc                = 0.0229038661772,
            proximal_init_dist          = (0.29593870493169883, 0.472835091999874),
            proximal_potential_pool     = 254,
            proximal_segments           = 9,
            proximal_thresh             = 0.138576319558,
            target_sparsity             = 0.00740525152346,)
        self.l23_radii = ()
        self.l4 = unified.UnifiedParameters(
            AP                          = 161,
            boosting_alpha              = 0.000719740044508,
            cells                       = 3752,
            distal_add_synapses         = 13,
            distal_init_dist            = (0.23844057563900786, 0.06096389011569249),
            distal_initial_segment_size = 18,
            distal_learning_threshold   = 6,
            distal_mispredict_dec       = 0.00125705924021,
            distal_permanence_dec       = 0.00782723575556,
            distal_permanence_inc       = 0.0472237864003,
            distal_permanence_thresh    = 0.136704751456,
            distal_predicted_boost      = 6,
            distal_predictive_threshold = 13,
            distal_segments_per_cell    = 9,
            distal_synapses_per_segment = 39,
            min_stability               = None,
            predicted_proximal_boost    = 0.313910585562,
            proximal_active_thresh      = 7,
            proximal_add_synapses       = 0,
            proximal_dec                = 0.00614680971289,
            proximal_inc                = 0.0229771432868,
            proximal_init_dist          = (0.13862464334617058, 0.084322450006509356),
            proximal_potential_pool     = 833,
            proximal_segments           = 4,
            proximal_thresh             = 0.17543370616,
            target_sparsity             = 0.0402221422177,)
        self.l4_radii = ()
        self.mutation_rate = 0.0620711317884
        self.sdrc = classifiers.SDRC_Parameters(
            alpha = 0.000685079420787,)

    def evaluate(self, debug):
        # Setup test and train datasets and perform lexical analysis.  First get
        # full text of training dataset into a string.
        if self.dataset == 'gutenberg':
            text_stream = read_corpus(debug = debug)
        elif self.dataset == 'states':
            text_stream = state_name_reader(debug = debug)
        train_dataset = []
        for i in range(self.train_time):
            char = next(text_stream)
            train_dataset.append(char)
        train_dataset   = ''.join(train_dataset)
        # Search for words in the dataset.  Store the words as keys in a
        # histogram of word occurances.
        word_regex = r"\w(\w|')*"
        word_iter  = re.finditer(word_regex, train_dataset)
        word_hist  = {}
        # train_word_spans stores where each word is located, list of pairs of
        # (start, end) index into train_dataset.
        train_word_spans = []
        for match in word_iter:
            span = match.span()     # Returns pair of (start-index, end-index)
            word = train_dataset[span[0] : span[1]]
            if word not in word_hist:
                word_hist[word] = 0
            word_hist[word] += 1
            train_word_spans.append(span)
        # Sort words by the number of times the occur in the train_dataset.
        # Break ties randomly.
        word_list = list(word_hist.keys())
        word_freq = [-(word_hist[word] + random.random()) for word in word_list]
        word_rank = np.take(word_list, np.argsort(word_freq))
        # Get word_list and word_freq out of memory, from here on use word_rank & word_hist.
        word_list = None; word_freq = None
        # Select some common words to test vocabulary with.
        test_words = word_rank[: self.test_words].tolist()
        # Assign each vocabulary word an integer identifier.  A test words
        # identifier doubles as its index into the test_words list.
        if False:
            test_words.sort()   # Make the index easier for humans to use.
        # The first entry is special B/C when the SDRC can't identify the word
        # at all, it outputs all zeros.  Then np.argmax() outputs as index 0 as
        # the best prediction.
        test_words.insert(0, "WORD_UNKNOWN")
        word_hist[test_words[0]] = 0
        test_word_id_lookup = {word : index for index, word in enumerate(test_words)}
        # Search for examples of the vocabulary words used in sentances.  First
        # read a large amount of sample text.  Only relevant sections of the
        # test_dataset are used, the ranges of text are stored in variable
        # test_sentance_spans.
        test_dataset = []
        for i in range(int(self.test_time)):
            char = next(text_stream)
            test_dataset.append(char)
        test_dataset        = ''.join(test_dataset)
        word_iter           = re.finditer(word_regex, test_dataset)
        # The following two lists hold pairs of (start, end) slice indexes into
        # test_dataset.  They are NOT the same length because overlapping
        # test_sentance_spans are merged into a single example containing
        # several of the vocabulary words.
        test_word_spans     = []    # Spans of just the test vocabulary words.
        test_sentance_spans = []    # Spans of test words with preceding context included.
        test_hist           = {word: 0 for word in test_words}
        for match in word_iter:
            span       = match.span()     # Returns pair of (start-index, end-index)
            start, end = span
            word       = test_dataset[start : end]
            if word not in test_word_id_lookup.keys():
                continue
            # Ignore test vocabulary words after they've been seen many times.
            if test_hist[word] >= self.test_sample:
                continue
            test_hist[word] += 1
            test_word_spans.append(span)
            context_start = max(0, start - self.min_context)
            if test_sentance_spans and test_sentance_spans[-1][1] >= context_start:
                # Extend the last test sentance and test this additional word using it.
                context_start           = test_sentance_spans[-1][0]
                test_sentance_spans[-1] = (context_start, end)
            else:
                # Add a new test sentance.
                test_sentance_spans.append((context_start, end))
        len_test_dataset = sum(e - s for s, e in test_sentance_spans)
        if debug:
            print('Training dataset size:', self.train_time, 'characters,', len(train_word_spans), 'words,', len(word_hist), 'unique words.')
            print('Test vocabulary size:', len(test_words), 'words.')
            min_freq = min(word_hist[word] for word in test_words[1:])
            max_freq = max(word_hist[word] for word in test_words[1:])
            print('Test vocabulary samples:', ', '.join(random.sample(test_words[1:], 6)) + '.')
            print('Test vocabulary min & max occurances in training dataset: %d - %d.'%(min_freq, max_freq))
            test_hist_values = list(test_hist[word] for word in test_words[1:])
            min_freq = min(test_hist_values)
            avg_freq = np.mean(test_hist_values)
            max_freq = max(test_hist_values)
            print('Test vocabulary min/mean/max occurances in testing dataset: %d / %.1f / %d.'%
                (min_freq, avg_freq, max_freq))
            print('Test dataset size:', len_test_dataset, 'characters,', len(test_word_spans), 'vocabulary words.')
            print('Test sentance average length: %.1f characters.'%(len_test_dataset/len(test_sentance_spans)))
            if self.list_test_words:
                print('Index) Word, Train samples, Test samples.')
                if False:
                    # Sort by number of samples in dataset.
                    # TODO: This would be more useful if it sorted the actual test_words list.
                    test_freq = [-test_hist[word] for word in test_words]
                    test_rank = np.take(test_words, np.argsort(test_freq))
                    ordered_words = test_rank
                else:
                    # Sort by index.
                    ordered_words = test_words
                # Print index of test words.
                for word in ordered_words:
                    index         = test_word_id_lookup[word]
                    train_samples = word_hist[word]
                    test_samples  = test_hist[word]
                    fmt_str = '%3d) %-15s\t%2d, %2d'
                    print(fmt_str%(index, word, train_samples, test_samples))
                if True:
                    # Look at some test sentances
                    sample_spans     = random.sample(test_sentance_spans, min(10, len(test_sentance_spans)))
                    sample_sentances = [test_dataset[s[0]: s[1]] for s in sample_spans]
                    print('Sample test sentances:\n\t', '\n\n\t'.join(sample_sentances))
            print()
        # After seeing all of the words in either dataset, wait forever for the
        # next word, or until the dataset is finished playing.  This case is
        # needed when the dataset ends with white space.
        train_word_spans.append((float('inf'), float('inf')))
        test_word_spans.append((float('inf'), float('inf')))
        # if len(test_words) != self.test_words + 1:
        #     raise ValueError('Could not find %d test words'%self.test_words)

        # Setup AI.
        timer = genetics.speed_fitness(self.time_limit / 3, self.time_limit)
        enc = encoders.EnumEncoder(self.enc_bits, self.enc_sparsity, diag=False)

        # Make the context SDR which both L4 and L23 use to predict the future.
        context_size = self.l4.cells + self.l23.cells
        context      = SDR((context_size,))
        l4  = unified.Unified(self.l4,
                             input_sdr     = enc.output_sdr,
                             context_sdr   = context,
                             macro_columns = (1,),
                             radii         = self.l4_radii,)
        l23 = unified.Unified(self.l23,
                             input_sdr     = l4.active,
                             context_sdr   = context,
                             macro_columns = (1,),
                             radii         = self.l23_radii,)

        l4_sdrc  = classifiers.SDR_Classifier(self.sdrc, l4.active,  (len(test_words),), 'index')
        l23_sdrc = classifiers.SDR_Classifier(self.sdrc, l23.active, (len(test_words),), 'index')

        def reset():
            l4.reset()
            l23.reset()

        def compute(learn=True):
            context.assign_flat_concatenate([l4.active, l23.active])
            if self.l4_only:
                # Test L4 in isolation, Disable feedback from L2/3 to L4.
                zeros_like_l23 = SDR(l23.active); zeros_like_l23.zero()
                context.assign_flat_concatenate([l4.active, zeros_like_l23])
            l4.compute()
            l23.compute()
            if learn:
                l4.learn()
                l23.learn()

        if debug:
            print('SDR DEBUG:', sdr.debug)
            if self.l4_only:
                print("L4 Isolated, Disabled L2/3 -> L4 Feedback.")
            if False:
                print('L4', l4.statistics())
                print('L23', l23.statistics())

        # Train by reading books.
        if self.train_no_stability:
            self.l23.min_stability = 0
            assert(debug)
            print('L23 min stability set to', self.l23.min_stability)
        if debug:
            print('Training ...')
        word            = None  # Current word or None, AI trains to predict this variable.
        word_index      = None  # Index of current word in test_data, or None if its not a test word.
        word_span_index = 0     # Index of current word in train_dataset
        reset()
        for step in range(self.train_time):
            # Determine the current word.
            start, end = train_word_spans[word_span_index]
            if step == start:
                word = train_dataset[start : end]
                try:
                    word_index = (test_word_id_lookup[word],)
                except KeyError: # Word is not in vocabulary test words, SDRC should ignore it.
                    word_index = None
            if step == end:
                word = None
                word_index = None
                word_span_index += 1
            # Process the next letter of the book.
            char = train_dataset[step]
            enc.encode(char)
            compute(learn=True)
            if word_index is not None and step == end-1:
                l4_sdrc.train(input_sdr=None, out=word_index)
                l23_sdrc.train(input_sdr=None, out=word_index)

        # Test.  Measure:
        # 1) Stability,
        # 2) Anomaly,
        # 3) Word recognition accuracy and cross-catagory confusion.
        real_min_stab = l23.args.min_stability
        if self.test_no_stability:
            l23.args.min_stability = 0
        if debug:
            print('Testing ...')
            if l23.args.min_stability != real_min_stab:
                print('L23 min stability changed to', l23.args.min_stability)
            else:
                print('L23 min stability remains at', l23.args.min_stability)
        l23_stability = 0.  # Accumulates the L2/3 stability.
        l4_anomaly    = 0.  # Accumulates the L4 anomaly.
        l23_anomaly   = 0.  # Accumulates the L2/3 anomaly.
        l4_accuracy   = 0.  # Accumulates the L4 word classificatioon accuracy.
        l23_accuracy  = 0.  # Accumulates the L2/3 word classificatioon accuracy.
        max_accuracy  = 0.  # Number of samples accumulated in variable 'l23_accuracy'.
        l4_end_accuracy  = 0.   # Like 'l4_accuracy' but only measured on the final letter of the word.
        l23_end_accuracy = 0.   # Like 'l23_accuracy' but only measured on the final letter of the word.
        max_end_accuracy = 0.   # Number of samples accumulated in variable 'l23_end_accuracy'.
        l23_confusion    = np.zeros((len(test_words), len(test_words)))
        l4_confusion     = np.zeros((len(test_words), len(test_words)))
        next_span_index = 0 # Index of current word in test_word_spans (or next word if not currently on a word).
        for sentance_start, sentance_end in test_sentance_spans:
            reset()
            word_index = None   # Index of current word, or None.
            for index in range(sentance_start, sentance_end):
                # Determine the current word.  Allow words to immediately follow
                # each other, they in case they're seperated by a reset and zero
                # characters of context.
                word_start, word_end = test_word_spans[next_span_index]
                if index >= word_end:
                    word_index = None
                    next_span_index += 1
                word_start, word_end = test_word_spans[next_span_index]
                if index >= word_start:
                    word = test_dataset[word_start : word_end]
                    word_index = test_word_id_lookup[word]
                # Process the current character.
                char = test_dataset[index]
                enc.encode(char)
                compute(learn=False)
                # Measure.
                if real_min_stab > 0:
                    l23_stability += min(l23.stability, real_min_stab) / real_min_stab
                else:
                    l23_stability += 1
                l4_anomaly  += l4.anomaly
                l23_anomaly += l23.anomaly
                if word_index is not None:
                    l4_prediction  = l4_sdrc.predict()
                    l23_prediction = l23_sdrc.predict()
                    l4_best_guess = np.argmax(l4_prediction)
                    l23_best_guess = np.argmax(l23_prediction)
                    if l23_best_guess == word_index:
                        l23_accuracy += 1
                        if index == word_end - 1:
                            l23_end_accuracy += 1
                    if l4_best_guess == word_index:
                        l4_accuracy  += 1
                        if index == word_end - 1:
                            l4_end_accuracy += 1
                    max_accuracy += 1
                    if index == word_end - 1:
                        max_end_accuracy += 1
                    # Update confusion matixes.  Prediction is a PDF, sum must equal 1.
                    if True:
                        l23_confusion[word_index, l23_best_guess] += 1
                        if index == word_end - 1:
                            l4_confusion [word_index, l4_best_guess]  += 1
                    else:
                        l23_prediction_sum = np.sum(l23_prediction)
                        if l23_prediction_sum != 0.:
                            l23_prediction /= l23_prediction_sum
                            l23_confusion[word_index, :] += l23_prediction
                        l4_prediction_sum = np.sum(l4_prediction)
                        if l4_prediction_sum != 0.:
                            l4_prediction /= l4_prediction_sum
                            l4_confusion[word_index, :] += l4_prediction
        # Divide all accumulators by the number of samples added to them.
        l23_stability    /= len_test_dataset
        l23_accuracy     /= max_accuracy
        l23_end_accuracy /= max_end_accuracy
        l23_anomaly      /= len_test_dataset
        l4_accuracy      /= max_accuracy
        l4_end_accuracy  /= max_end_accuracy
        l4_anomaly       /= len_test_dataset
        for label_idx, label in enumerate(test_words):
            # Divide by the number of PDF's which have accumulated at each
            # label, each PDF has sum of 1.
            l23_num_samples = np.sum(l23_confusion[label_idx, :])
            if l23_num_samples != 0:
                l23_confusion[label_idx, :] /= l23_num_samples
            l4_num_samples = np.sum(l4_confusion[label_idx, :])
            if l4_num_samples != 0:
                l4_confusion[label_idx, :] /= l4_num_samples

        def plot_sentance_stability(string):
            plt.figure('Stability')
            plt.ylim(-0.01, 1.01)
            plt.xlim(-0.5,  len(string) - 0.5)
            plt.xlabel('Time')
            plt.ylabel('L2/3 Overlap')
            plt.axhline(real_min_stab)
            stability  = []
            confidence = []
            anomaly    = []
            reset()
            for step, char in enumerate(string):
                enc.encode(char)
                compute(learn=False)
                stability.append(l23.stability)
                anomaly.append(l23.anomaly)
                prediction = l23_sdrc.predict()
                best_guess = test_words[np.argmax(prediction)]
                confidence.append(np.max(prediction) / np.sum(prediction))
                # 
                plt.axvline(step + .5, color='grey', alpha=0.25)
                plt.text(step - 0.25, .98, char)
                if char.isalpha():
                    plt.text(step - 0.25, 0.95, best_guess, rotation='vertical')
                # TODO: Determine which steps it learns on by looking at the dataset.
                elif step-1 >= 0 and string[step-1].isalpha():
                    plt.axvspan(step - 1.5, step - .5, color='yellow', alpha=0.5)
            plt.axvspan(step - .5, step + .5, color='yellow', alpha=0.5)
            plt.plot(np.arange(len(string)), stability, 'r-',)
                     # np.arange(len(string)), confidence,   'b-',)
                     # np.arange(len(string)), anomaly,   'b-',)
            # plt.title('L2/3 Overlap is Red,  Confidence is Blue')
            # plt.title('L2/3 Overlap is Red,  Anomaly is Blue')
            plt.title(('Top: Input Letter, Middle: Best Guess,\n' +
                'Bottom Graph: Red Line L2/3 Stability, Blue Line: Target Stability, Learning Enabled on Yellow Steps.'))

        # Report.
        fitness = {
            'L23_stability':    l23_stability,
            'L23_accuracy':     l23_accuracy,
            'L23_end_accuracy': l23_end_accuracy,
            'L23_anomaly':      l23_anomaly,
            'L4_accuracy':      l4_accuracy,
            'L4_end_accuracy':  l4_end_accuracy,
            'L4_anomaly':       l4_anomaly,
            'speed':            timer.done(),
            'memory':           genetics.memory_fitness(2e9, 3e9),
        }
        if debug:
            print()
            print('L4',  l4.statistics())
            print('L23', l23.statistics())

            span     = random.choice(test_sentance_spans)
            sentance = test_dataset[span[0] : span[1]]
            sentance = sentance[-100 :]    # Don't show too much text in one figure.
            if self.show_typos:
                sentance = ' '.join([mutate_word(w) for w in sentance.split(' ')])
            plot_sentance_stability(sentance)

            plt.figure('L23 Confusion Matrix')
            plt.imshow(l23_confusion, interpolation='nearest')
            plt.xlabel('Prediction')
            plt.ylabel('Label')
            plt.figure('L4 Confusion Matrix')
            plt.imshow(l4_confusion, interpolation='nearest')
            plt.xlabel('Prediction')
            plt.ylabel('Label')

            plt.show()
        return fitness

if __name__ == '__main__':
    X      = ASCII_Experiment
    parser = genetics.ExperimentMain.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='gutenberg',
            choices=['gutenberg', 'states',])

    parser.add_argument('-t', '--time', type=int,
        help='Number of characters of training data to use.')

    parser.add_argument('-w', '--words', type=int,
        help='Number of words to test it against.')

    parser.add_argument('--vocabulary', action='store_true',
        help='List all of the vocabulary test words.')

    parser.add_argument('--train_no_stability', action='store_true',
        help='Set L23.min_stability to zero durring training.')

    parser.add_argument('--test_no_stability', action='store_true',
        help='Set L23.min_stability to zero durring testing.')

    parser.add_argument('--typo', action='store_true',
        help='Insert typos into the testing data.')

    parser.add_argument('--L4_only', action='store_true',
        help='Isolate L4, disable feedback from L2/3 to L4.')

    args = parser.parse_args()

    X.list_test_words    = args.vocabulary
    X.test_no_stability  = args.test_no_stability
    X.train_no_stability = args.train_no_stability
    X.show_typos         = args.typo
    X.l4_only            = args.L4_only

    # Setup the dataset's parameters, sort out defaults & command line overrides.
    X.dataset = args.dataset
    if X.dataset == 'gutenberg':
        # Read from the 100(+) most popular works on Project Gutenburg.
        X.train_time  = args.time # if args.time is not None else 25 * 1000
        X.test_time   = 1 * X.train_time
        X.test_words  = args.words if args.words is not None else 1000
        X.min_context = 40    # Least amount of preceding text to show AI before each test word.
        X.test_sample = 10    # Ignore test dataset vocabulary words after they've been tested this many times.
        X.time_limit  = 180   # Minutes, enforced by a timer & interupt (except if debug=True).
    elif X.dataset == 'states':
        # Read from the states names dataset.
        X.train_time  = 10 * 1000
        X.test_time   = 10 * 1000
        X.test_words  = 52
        X.min_context = 40    # Least amount of preceding text to show AI before each test word.
        X.test_sample = 10    # Ignore test dataset vocabulary words after they've been tested this many times.
        X.time_limit  = 30    # Minutes, enforced by a timer & interupt (except if debug=True).
        assert(args.time is None)  # Not a valid argument for the states names dataset.
        assert(args.words is None) # Not a valid argument for the states names dataset.

    genetics.ExperimentMain(X, args)
