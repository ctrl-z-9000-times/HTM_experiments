#!/usr/bin/python3
# Written by David McDougall, 2017

from htm import *

def number_test():
    """
    Test the Satial Pooler by encoding some integers and observing the results.
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
        inp_a = 765
        sdr_a = machine.compute(encoder.encode(inp_a), learn=False)
        orig = set(zip(*sdr_a))
        for x in range(0, on):
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

    # Train
    train_time = len(data_set)
    for i in range(train_time):
        if True:
            inp = random.choice(data_set)
            inp_enc = encoder.encode(inp)
        else:
            # Tests w/ Random Inputs
            inp_enc = np.random.random(encoder.output_shape) > .5
        state = machine.compute(inp_enc)

        modulus = train_time // 10
        if i % modulus == modulus-1:
            machine.entropy()

        # Halftime evaluation
        if i == train_time//2:
            sep_mid, ovlp_mid = overlap_vs_input_dist()
            rob_mid = machine.noise_robustness(enc_data)

    # Evaluate
    print('duty cycle min ',  np.min(machine.average_activations)  *100,'%')
    print('duty cycle mean',  np.mean(machine.average_activations) *100,'%')
    print('duty cycle std ',  np.std(machine.average_activations)  *100,'%')
    print('duty cycle max ',  np.max(machine.average_activations)  *100,'%')

    sep, ovlp = overlap_vs_input_dist()
    rob_after = machine.noise_robustness(enc_data)

    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(sep_b4, ovlp_b4, 'r', sep, ovlp, 'g', sep_mid, ovlp_mid, 'b')
    plt.title('Overlap vs Input Distance, \nRed is before, Blue is halfway, Green is after training %d cycles'%train_time)

    plt.subplot(2, 1, 2)
    plt.plot(rob_before[0], rob_before[1], 'r', rob_after[0], rob_after[1], 'g', rob_mid[0], rob_mid[1], 'b')
    plt.title('Overlap vs Input Noise')
    plt.show()

number_test()
