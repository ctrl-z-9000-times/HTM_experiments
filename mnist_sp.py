#!/usr/bin/python3
# Written by David McDougall, 2017


from htm import *
import time


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
                vec = np.array(vec, dtype=np.float32)
                buf = np.reshape(vec, (rows, cols, 1))
                imgs.append(buf)
            assert(len(raw) == data_start + img_size * num_imgs)   # All data should be used.
        return imgs

    train_labels = load_labels('MNIST_data/train-labels-idx1-ubyte.gz')
    train_images = load_images('MNIST_data/train-images-idx3-ubyte.gz')
    test_labels  = load_labels('MNIST_data/t10k-labels-idx1-ubyte.gz')
    test_images  = load_images('MNIST_data/t10k-images-idx3-ubyte.gz')

    return train_labels, train_images, test_labels, test_images


def synthesize(seed, diag=False):
    """
    Modify an image with random shifts, scales, and rotations.
    Use this function to expand the training dataset and make it more robust to these transforms.

    TRANSLATION DISABLED!
    MNIST doesn't have any shifts in it so adding them in makes the task harder.

    TODO: Stretching/scaling/skewing images
    TODO: Consider adding gausian noise
    """
    # Apply a random rotation
    theta_max = 15      # degrees
    theta = random.uniform(-theta_max, theta_max)
    synth = scipy.ndimage.interpolation.rotate(seed, theta, order=0, reshape=False)

    # Stretch the image in a random direction
    pass    # TODO

    if False:
        # Find the bounding box of the character
        r_occupied = np.sum(synth, axis=1)
        for r_min in range(len(r_occupied)):
            if r_occupied[r_min]:
                break
        for r_max in range(len(r_occupied)-1, -1, -1):
            if r_occupied[r_max]:
                break

        c_occupied = np.sum(synth, axis=0)
        for c_min in range(len(c_occupied)):
            if c_occupied[c_min]:
                break
        for c_max in range(len(c_occupied)-1, -1, -1):
            if c_occupied[c_max]:
                break

        # Apply a random shift
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


def MNIST_test(r, t):

    # Load and prepare the data
    train_labels, train_images, test_labels, test_images = load_mnist()
    if False:
        # Experiment to verify that input dimensions are handled correctly
        # If you enable this, don't forget to rescale the radii as well as the input.
        from scipy.ndimage import zoom
        new_sz = (1, 4, 1)
        train_images = [zoom(im, new_sz, order=0) for im in train_images]
        test_images  = [zoom(im, new_sz, order=0) for im in test_images]
    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))

    start_time = time.time()
    enc = ImageEncoder(train_images[0].shape[:2])
    print("Input Shape", enc.output_shape)
    # col_shape = (28, 28)
    # col_shape = (56, 56)
    col_shape = (112, 112)
    print("Column Shape", col_shape)
    radii = (r, r)
    print("Radii", radii)
    machine = SpatialPooler(enc.output_shape, col_shape, radii)

    # Make an SDR Maximum Likelyhood classifier
    class_shape = (10,)
    sdrc = SDR_Classifier(col_shape, class_shape, None)
    # sdrc = KNN_Classifier(col_shape, class_shape)

    plot_noise_robustness = False
    rand_imgs     = random.sample(test_images, 100)
    rand_imgs_enc = [enc.encode(np.squeeze(q)) for q in rand_imgs]
    if plot_noise_robustness:
        x0, y0 = machine.noise_robustness(rand_imgs_enc)


    print("Initialiation complete, Begining training phase...")
    # The difference between x1 and x100 the training time is 79.86% and 81.19% accuracy...
    # These things might be immune to overtraining.
    train_cycles = int(round(len(train_images) * t))
    compute_time = 0
    print("Training Time", train_cycles)
    for i in range(train_cycles):
        img, lbl = random.choice(training_data)
        img = synthesize(img, diag=False)       # SYNTHETIC TRAINING DATA
        compute_start = time.time()             # Includes time to encode input
        img_enc = enc.encode(np.squeeze(img))
        state = machine.compute(img_enc)
        sdrc.train(state, (lbl,))
        compute_time += time.time() - compute_start
        modulus = train_cycles // 10
        if i % modulus == modulus-1:
            machine.entropy()

    print("Training complete, Begining evaluation phase...")

    print('duty cycle min ',  np.min(machine.average_activations) * 100, '%')
    print('duty cycle mean', np.mean(machine.average_activations) * 100, '%')
    print('duty cycle std ',  np.std(machine.average_activations) * 100, '%')
    print('duty cycle max ',  np.max(machine.average_activations) * 100, '%')

    # Evaluate the classifier
    score = 0
    all_pred = set()
    for img, lbl in test_data:
        if True:
            img_enc = np.squeeze(enc.encode(img))
            state = machine.compute(img_enc, learn=False)
            prediction = np.argmax(sdrc.predict(state))
        else:
            # Test Random Performance
            prediction = np.argmax(np.random.random(class_shape))
        all_pred.add(prediction)
        if prediction == lbl:
            score += 1
    print("Score", score, '/', len(test_data))
    print("Predicted Classes", all_pred)    # Sanity check
    end_time = time.time()
    print("Compute time %g seconds"%(compute_time / train_cycles))
    print("Elapsed time %g seconds"%(end_time - start_time))

    # Show Diagnostics for a sample input
    # First run a sample input through the pipeline to setup the debug variables.
    state = machine.compute(rand_imgs_enc[0])   # Learning enabled + boosting
    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(np.dstack([rand_imgs[0]]*3)/255., interpolation='nearest')
    plt.title("Input Image")

    plt.subplot(2, 3, 2)
    plt.imshow(machine.zz_raw, interpolation='nearest')
    plt.title('Raw Excitement, radius' + str(radii))

    plt.subplot(2, 3, 3)
    plt.imshow(machine.average_activations.reshape(col_shape), interpolation='nearest')
    plt.title('Average Duty Cycle (alpha = %g)'%machine.average_activations_alpha)

    plt.subplot(2, 3, 4)
    plt.imshow(machine.zz_boostd.reshape(col_shape), interpolation='nearest')
    plt.title('Boosted')

    plt.subplot(2, 3, 5)
    plt.imshow(machine.zz_norm, interpolation='nearest')
    plt.title('Locally Inhibited Excitement')

    plt.subplot(2, 3, 6)
    active_state_visual = np.zeros(col_shape)
    active_state_visual[state] = 1
    plt.imshow(np.dstack([active_state_visual]*3), interpolation='nearest')
    plt.title("Active Columns (%d train cycles)"%machine.age)

    # Plot Robustness index
    if plot_noise_robustness:
        x1, y1 = machine.noise_robustness(rand_imgs_enc)
        plt.figure(2)
        plt.plot(x0, y0, 'r', x1, y1, 'g')
        plt.title("Robustness. Red is before, Green is after training %d cycles"%machine.age)

    # Show a table of SP inputs & outputs
    if False:
        examples = 5    # This many rows of examples, one example per row
        cols = 4        # This many columns
        plt.figure(3)
        for row in range(examples):
            for sub_col in range(int(cols / 2)):
                img, lbl = random.choice(test_data)
                img_enc = np.squeeze(enc.encode(img))
                state = machine.compute(img_enc, learn=False)   # No boosting here!
                plt.subplot(examples, cols, row*cols + sub_col*2 + 1)
                plt.imshow(np.dstack([img]*3)/255., interpolation='nearest')
                plt.title("Label: %s"%lbl)
                # Show the column activations
                state_visual = np.zeros(col_shape)
                state_visual[state] = 1
                plt.subplot(examples, cols, row*cols + sub_col*2 + 2)
                plt.imshow(np.dstack([state_visual]*3), interpolation='nearest')
                plt.title("Active Columns")

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', type=float, default=3)
    parser.add_argument('-t', '--time', type=float, default=1.0)
    parser.add_argument('--note', type=str)
    args = parser.parse_args()

    if args.note:
        print()
        print(args.note)
    MNIST_test(r=args.radius, t=args.time)
