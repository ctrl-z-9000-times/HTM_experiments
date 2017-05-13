#!/usr/bin/python3

from htm import *

def number_test():
    """
    Encode some numbers to test the spacial pooler.
    """
    res = 1
    size = 501
    on = 50
    data_set = range(10000)
    encoder = RandomDistributedScalarEncoder(res, size, on)
    col_shape = (1024,)
    machine = SpatialPooler(encoder.output_shape, col_shape)

    enc_data = [encoder.encode(z) for z in random.sample(data_set, 100)]
    rob_before = machine.noise_robustness(enc_data)

    def overlap_vs_input_dist():
        sep = []
        ovlp = []
        inp_a = 333
        sdr_a = machine.compute(encoder.encode(inp_a), learn=False)
        orig = set(zip(*sdr_a))
        for x in range(0, 75):
            inp_b = inp_a + x
            sdr_b = machine.compute(encoder.encode(inp_b), learn=False)

            # Calculate the overlap in SP output.
            near = set(zip(*sdr_b))
            overlap = len(orig.intersection(near))
            overlap_pct = overlap / len(orig)
            sep.append(x)
            ovlp.append(overlap_pct)
        return sep, ovlp

    sep_b4, ovlp_b4 = overlap_vs_input_dist()

    # Training to to recognise some numbers.
    train_time = 6 * len(data_set)
    for i in range(train_time):
        inp = random.choice(data_set)
        inp_enc = encoder.encode(inp)
        # inp_enc = np.random.random(encoder.output_shape) > .5     # Tests w/ Random Inputs
        state = machine.compute(inp_enc)

        modulus = train_time // 20
        if i % modulus == modulus-1:
            machine.entropy()

    # Evaluate
    sep, ovlp = overlap_vs_input_dist()
    rob_after = machine.noise_robustness(enc_data)

    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(sep_b4, ovlp_b4, 'r', sep, ovlp, 'b')
    plt.title('Overlap vs Input Distance, Red is before, Green is after training %d cycles'%train_time)

    plt.subplot(1, 2, 2)
    plt.plot(rob_before[0], rob_before[1], 'r', rob_after[0], rob_after[1], 'b')
    plt.title('Overlap vs Input Noise')
    plt.show()

number_test()
