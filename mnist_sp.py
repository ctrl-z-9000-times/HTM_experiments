#!/usr/bin/python3
# Written by David McDougall, 2017

from htm import *
import genetics

def load_mnist():
    """See: http://yann.lecun.com/exdb/mnist/ for MNIST download and binary file format spec."""
    import gzip
    import numpy as np

    def int32(b):
        i = 0
        for char in b:
            i *= 256
            # i += ord(char)    # python2?
            i += char
        return i

    def load_labels(file_name):
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2049)  # Magic number
            labels = []
            for char in raw[8:]:
                # labels.append(ord(char))      # python2?
                labels.append(char)
        return labels

    def load_images(file_name):
        with gzip.open(file_name, 'rb') as f:
            raw = f.read()
            assert(int32(raw[0:4]) == 2051)    # Magic number
            num_imgs   = int32(raw[4:8])
            rows       = int32(raw[8:12])
            cols       = int32(raw[12:16])
            assert(rows == 28)
            assert(cols == 28)
            img_size   = rows*cols
            data_start = 4*4
            imgs = []
            for img_index in range(num_imgs):
                vec = raw[data_start + img_index*img_size : data_start + (img_index+1)*img_size]
                # vec = [ord(c) for c in vec]   # python2?
                vec = list(vec)
                vec = np.array(vec, dtype=np.uint8)
                buf = np.reshape(vec, (rows, cols, 1))
                imgs.append(buf)
            assert(len(raw) == data_start + img_size * num_imgs)   # All data should be used.
        return imgs

    train_labels = load_labels('MNIST_data/train-labels-idx1-ubyte.gz')
    train_images = load_images('MNIST_data/train-images-idx3-ubyte.gz')
    test_labels  = load_labels('MNIST_data/t10k-labels-idx1-ubyte.gz')
    test_images  = load_images('MNIST_data/t10k-images-idx3-ubyte.gz')

    return train_labels, train_images, test_labels, test_images


# TODO: Synthesize should randomly stretch/scale/skew images.
def synthesize(seed, diag=False):
    """
    Modify an image with random shifts, scales, and rotations.
    Use this function to expand the training dataset and make it more robust to these transforms.

    Note: translation is worse for training MNIST b/c the test set is centered.
    Translation just makes the problem harder.

    TODO: Stretching/scaling/skewing images
    """
    # Apply a random rotation
    theta_max = 15      # degrees
    theta = random.uniform(-theta_max, theta_max)
    synth = scipy.ndimage.interpolation.rotate(seed, theta, order=0, reshape=False)

    def bounding_box(img):
        # Find the bounding box of the character
        r_occupied = np.sum(img, axis=1)
        for r_min in range(len(r_occupied)):
            if r_occupied[r_min]:
                break
        for r_max in range(len(r_occupied)-1, -1, -1):
            if r_occupied[r_max]:
                break

        c_occupied = np.sum(img, axis=0)
        for c_min in range(len(c_occupied)):
            if c_occupied[c_min]:
                break
        for c_max in range(len(c_occupied)-1, -1, -1):
            if c_occupied[c_max]:
                break
        return r_min, r_max, c_min, c_max

    # Stretch the image in a random direction
    pass

    if False:
        # Apply a random shift
        r_min, r_max, c_min, c_max = bounding_box(synth)
        r_shift = random.randint(-r_min, len(r_occupied) -1 -r_max)
        c_shift = random.randint(-c_min, len(c_occupied) -1 -c_max)
        synth = scipy.ndimage.interpolation.shift(synth, [r_shift, c_shift, 0])

    if diag:
        from matplotlib import pyplot as plt
        plt.figure(1)
        sz = 3
        example_synths = [synthesize(seed, diag=False) for _ in range(sz**2 - 2)]
        example_synths.append(synth)
        plt.subplot(sz, sz, 1)
        plt.imshow(np.dstack([seed/255]*3), interpolation='nearest')
        plt.title("Seed")
        for i, s in enumerate(example_synths):
            plt.subplot(sz, sz, i+2)
            plt.imshow(np.dstack([s/255]*3), interpolation='nearest')
            plt.title("Synthetic")
        plt.show()

    return synth


class MNIST_Parameters(genetics.Parameters):
    parameters = ['sp', 'sdrc',]
    def __init__(self,):
        self.sp = SpatialPoolerParameters(
            column_dimensions   = (1.216e+02, 1.274e+02),
            radii               = (3.308e+00, 1.933e+00),
            potential_pool      = 1.173e+02,
            sparsity            = 1.047e-02,
            coincidence_inc     = 3.532e-02,
            coincidence_dec     = 1.069e-02,
            permanence_thresh   = 3.901e-01,
            boosting_alpha      = 7.503e-04,
        )
        self.sdrc = SDRC_Parameters(alpha=1.129e-03)


def evaluate(parameters):
    # Load data, Setup spatial pooler machine.
    train_labels, train_images, test_labels, test_images = load_mnist()
    training_data = list(zip(train_images, train_labels))
    test_data     = list(zip(test_images, test_labels))
    enc           = BWImageEncoder(train_images[0].shape[:2], diag=False)
    machine       = SpatialPooler(parameters.sp, enc.output_shape)
    class_shape   = (10,)
    sdrc          = SDR_Classifier(parameters.sdrc, machine.column_dimensions, class_shape, 'index')

    # Training Loop
    train_cycles = len(train_images) * 1/2
    for i in range(int(round(train_cycles))):
        img, lbl      = random.choice(training_data)
        img           = synthesize(img, diag=False)
        img_enc       = enc.encode(np.squeeze(img))
        state         = machine.compute(img_enc)
        sdrc.train(state, (lbl,))

    # Testing Loop
    score = 0
    for img, lbl in test_data:
        img_enc     = np.squeeze(enc.encode(img))
        state       = machine.compute(img_enc, learn=False)
        prediction  = np.argmax(sdrc.predict(state))
        if prediction == lbl:
            score   += 1
    return score / len(test_data)


if False:
    # I'm keeping the following diagnostic code snippets just in case I ever
    # need them.  They are old and may not work.
    from matplotlib import pyplot as plt

    if False:
        # Experiment to test what happens when areas are not given meaningful
        # input.  Adds 2 pixel black border around image.  Also manually
        # disabled translation in the synthesize funtion.
        def expand_images(mnist_images):
            new_images = []
            for img in mnist_images:
                assert(img.shape == (28, 28, 1))
                new_img = np.zeros((32, 32, 1))
                new_img[2:-2, 2:-2, :] = img
                new_images.append(new_img)
            return new_images
        train_images = expand_images(train_images)
        test_images  = expand_images(test_images)

    if False:
        # Experiment to verify that input dimensions are handled correctly
        # If you enable this, don't forget to rescale the radii as well as the input.
        from scipy.ndimage import zoom
        new_sz = (1, 4, 1)
        train_images = [zoom(im, new_sz, order=0) for im in train_images]
        test_images  = [zoom(im, new_sz, order=0) for im in test_images]

    if False:
        # Show Diagnostics for a sample input
        state = machine.compute(rand_imgs_enc[0], diag=True)   # Learning & boosting enabled
        machine.proximal.synapse_histogram(diag=True)
        machine.proximal.permanence_histogram(diag=True)

        if plot_noise_robustness:
            x1, y1 = machine.noise_robustness(rand_imgs_enc)
            plt.figure(2)
            plt.plot(x0, y0, 'r', x1, y1, 'g')
            # plt.title("Noise Robustness. Red is before, Green is after training %d cycles"%machine.age)

    if False:
        # Show a table of SP inputs & outputs
        examples = 4    # This many rows of examples, one example per row
        cols = 6        # This many columns
        plt.figure(instance_tag + ' examples')
        for row in range(examples):
            for sub_col in range(int(cols / 2)):
                img, lbl = random.choice(test_data)
                img_enc = np.squeeze(enc.encode(img))
                state = machine.compute(img_enc, learn=False)   # No boosting here!
                prediction = np.argmax(sdrc.predict(state))
                plt.subplot(examples, cols, row*cols + sub_col*2 + 1)
                plt.imshow(np.dstack([img]*3)/255., interpolation='nearest')
                plt.title("Label: %s"%lbl)
                # Show the column activations
                state_visual = np.zeros(col_shape)
                state_visual[state] = 1
                plt.subplot(examples, cols, row*cols + sub_col*2 + 2)
                plt.imshow(np.dstack([state_visual]*3), interpolation='nearest')
                plt.title("Classification %d"%prediction)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--processes',  type=int, default=7, 
        help="Number of processes to use.")
    parser.add_argument('-p', '--population', type=int, default=50)
    parser.add_argument('-e', '--epochs',    type=int, default=1)
    parser.add_argument('--seed',        type=bool, default=False,
        help='If True, start a new population and ignore any saved checkpoints, '+
             'if False then loads the latest checkpoint from file.')
    parser.add_argument('--checkpoint',  type=str,  default='checkpoint',
        help='What name to save the results by.')
    args = parser.parse_args()

    genetics.genetic_algorithm(
        MNIST_Parameters,
        evaluate,
        population_size                 = args.population,
        num_epochs                      = args.epochs,
        seed                            = args.seed,
        seed_mutations_per_parameter    = 20,
        seed_mutation_percent           = 0.05,
        mutation_probability            = 0.20,
        mutation_percent                = 0.10,
        filename                        = args.checkpoint,
        num_processes                   = args.processes,
        profile                         = True,
    )
    exit(0)
