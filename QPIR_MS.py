#!/usr/bin/env python

from qiskit import *
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
import numpy as np
import itertools

provider = IBMQ.load_account()
api = provider._api
np.random.seed(0)


def int2bin(i, l): return np.fromstring(' '.join(bin(i)[2:].zfill(l)), dtype=np.int, sep=' ')


def bin2int(b): return b.dot(1 << np.arange(b.shape[-1] - 1, -1, -1, dtype=np.int))


def keywithmaxval(d):
    """ This function creates a list of the dict's keys and values and returns the key with the max value """
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def packer(db, b): return np.array(
    [(db[i] >> j) & 3 for i, j in itertools.product(range(db.size), range(0, b << 1, 2))])


def xordot(a, b, n):
    """ This function computes the scalar product in Z_2 """
    res = 0
    for i in range(n):
        res ^= a[i] * b[i]

    return res


def hamming_weight(int_no):
    """ This function computes the hamming weight of a integer number (seen as binary string) """
    c = 0
    while int_no:
        int_no &= (int_no - 1)
        c += 1

    return c


def bell_state(b):
    """ This function prepares the circuit and the max entangled states """
    n = int(b << 1)
    qc = QuantumCircuit(n, n)

    for i in range(0, n, 2):
        qc.h(i)
        qc.cx(i, i + 1)

    return qc


def server(f, m): return np.random.randint(0, m, size=f, dtype=np.int)


def query0(f): return np.random.randint(0, 2, size=f, dtype=bool)


def query1(q1, k):
    """ This function flips the k-th bit of a query """
    q2 = np.array(q1)
    q2[k] ^= 1
    return q2


def server_round(ddb, b, query, r, qc, label):
    """ Round of computation for each server
    Input:
        ddb = database (possibly packed) of files
        b = number of stripes into we divide the files = number of rounds to retrieve the file
        query = list of bits to request certain files from the database
        r = number of the round
        qc = bell state to modify
        label = label of the quantum system to modify
    h = scalar product of the query and the database of files
    """
    h = xordot(ddb[range(r, ddb.size, b)], query, query.size)

    if h >> 1:
        # If the first bit from the left is a 1, we apply an X gate
        qc.x(label)
    if h & 1:
        # If the second bit from the left is a 1, we apply a Z gate
        qc.z(label)


def server_computation(db, bit_n, query, qc, label):
    """ Computation for each server
    Input:
        db = database of files
        bit_n = number of bits in which we represent the files
        query = list of bits to request certain files from the database
        qc = bell state to modify
        label = label of the server to use
    b = number of stripes into we divide the files = number of rounds to retrieve the file
    In order to compute b, we divide the number of bits used to describe a file by 2, since we want to split the
    file in base 4, not in base 2. For example, if the number of bits used is 3, then 3 >> 1 = 1, so it wouldn't
    be right: we want the result of the division by excess, so we add 1 before the division.
    """
    b = np.int(bit_n + 1 >> 1)

    if b == 1:
        server_round(db, 1, query, 0, qc, label)
    else:
        ddb = packer(db, b)  # Packed database (with base 4)

        for r in range(b):
            server_round(ddb, b, query, r, qc, (r << 1) + label)


def measurement(qc, fp, shot=1024, device=False):
    """ Measurement of the quantum systems
    Input:
        qc = quantum systems to measure
        fp = file to print on the results
        shot = number of shots per measurement (max: 8192)
        device = preferred device for running the quantum measurement
    """
    for i in range(0, qc.n_qubits, 2):
        qc.cx(i, i + 1)
        qc.h(i)
        qc.measure([i, i + 1], [i, i + 1])

    try:
        if not device:
            device = least_busy(provider.backends(simulator=False,
                                                  filters=lambda x: x.configuration().n_qubits >= qc.n_qubits))
        print("Running on the following device: ", device)

        # Execute the job
        job_exp = execute(qc, backend=device, shots=shot)
        job_monitor(job_exp)
        res = job_exp.result()
        try:
            summary = api.get_job(job_exp._job_id)
            t = summary['summaryData']['resultTime']
        except:
            t = False

        print('\n ' + 50 * '%' + ' \n', file=fp)
        print("Running on the following device: ", device, file=fp)

        return [res.get_counts(qc), t]
    except:
        return [False, False]


def decision_maker(res, R, counter, g, lc, file_req_bin):
    """ Decision for the final measurement
    Input:
        res = dictionary of all the measurements from the quantum computer
        R = number of entangled states
        counter = list of counters (dict) for the shots in the number of trials in order to compute the average
                  shots per trial
        g = number of bits for making the majority decision (described in quantum_experiment)
        lc = list of counters of shots for the file requested per trial
        file_req_bin = file requested written in a binary string
    """
    dec = ''

    if g == 1:
        counter_list = [{'0': 0, '1': 0} for i in range(R << 1)]

        """ We increase the counter of shots per bit by reading all the measurements """
        for key in res.keys():
            for j in range(R << 1):
                """ counter and counter_list are lists of 2R dictionaries """
                counter[j][key[j]] += res[key]
                counter_list[j][key[j]] += res[key]

        """ We write the decision """
        for j in range(R << 1):
            dec += keywithmaxval(counter_list[j])

        """ We append to the list the number of shots for each bit of the requested file """
        lc.append(np.array([dd[bit] for dd, bit in zip(counter_list, file_req_bin)]))

    elif g == 2:
        counter_list = [{'00': 0, '01': 0, '10': 0, '11': 0} for i in range(R)]

        """ We increase the counter of shots per pair of bits by reading all the measurements """
        for key in res.keys():
            for j in range(R):
                """ counter and counter_list are lists of R dictionaries """
                counter[j][key[2 * j:2 * j + 2]] += res[key]
                counter_list[j][key[2 * j:2 * j + 2]] += res[key]

        """ We write the decision """
        for j in range(R):
            dec += keywithmaxval(counter_list[j])

        """ We append to the list the number of shots for each pair of bits of the requested file """
        lc.append(np.array([dd[pair]
                            for dd, pair in zip(counter_list, [file_req_bin[2*i:2*i+2] for i in range(R)])]))

    else:
        """ We append to the list the number of shots for the requested file """
        lc.append(res[file_req_bin])

        """ The decision is the key with max value in the dictionary of measurements """
        return keywithmaxval(res)

    """ We return the decision written in the cases g==1 or g==2 """
    return dec


def quantum_experiment(file_req, bit_n, trials, device):
    """ QPIR
    Initialize variables:
        Input:
        file_req = file to request to the server (zeros or ones, possibly)
        bit_n = number of bits to describe a file (min: 2; max: 14)
        trials = number of trials
    Other variables:
    m = number of files
    F = maximum file size
    I = index of file, 0, in order to request exactly the file we want
    shots = number of shots for measurements in each trial
    time = time taken on average per computation
    time_c = number of times "time" is actually measured
    ft = file on which we write the results of each trial
    fa = file on which we write the averages and the variances
    R = number of stripes into we divide the files = number of rounds to retrieve the file
        In order to compute b, we divide the number of bits used to describe a file by 2, since we want to split the
        file in base 4, not in base 2. For example, if the number of bits used is 3, then 3 >> 1 = 1, so it wouldn't
        be right: we want the result of the division by excess, so we add 1 before the division
    """
    m = 4096
    F = 1 << bit_n
    I = 0
    shots = 8192
    time = 0
    time_c = 0
    ft = open('methods' + str(bit_n) + '_' + str(file_req & 1) + '.txt', 'a+')
    fa = open('averages' + str(bit_n) + '_' + str(file_req & 1) + '.txt', 'a+')
    R = np.int(bit_n + 1 >> 1)

    file_req_bin = bin(file_req)[2:].zfill(R << 1)
    file_req_bin_v = int2bin(file_req, bit_n)

    """
    g = number of bits for making the majority decision:
        1:  majority decision for every bit
        2:  majority decision for pairs of bits
        3: majority decision among all the results
    We initialize a counter for each group, in the end we will take the average of the counts per group and we
    will compute the variance.
    We will store the number of shots taken for the requested file in lists (of counters) lc_g.
    We initialize also a counter for the number of errors for each bit, for each kind of decision. 
    We compute the accuracy for each kind of decision. """
    counter1 = [{'0': 0, '1': 0} for i in range(R << 1)]
    counter2 = [{'00': 0, '01': 0, '10': 0, '11': 0} for i in range(R)]
    lc_1 = []
    lc_2 = []
    lc_3 = []
    bit_errors_1 = np.zeros(R << 1, dtype=np.int)
    bit_errors_2 = np.zeros(R << 1, dtype=np.int)
    bit_errors_3 = np.zeros(R << 1, dtype=np.int)
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0

    """ Beginning of trials """
    trial = 0
    while trial < trials:
        """ Preparing the database and the qubits """
        db = np.append(file_req, server(m - 1, F))  # Database of files
        qc = bell_state(R)                          # Shared entangled states

        """ Query and computation for server 0 """
        q0 = query0(m)
        server_computation(db, bit_n, q0, qc, 0)

        """ Query and computation for server 1 """
        q1 = query1(q0, I)
        server_computation(db, bit_n, q1, qc, 1)

        """ Measurement of the resulting state """
        res, t = measurement(qc, ft, shot=shots, device=device)
        if res:
            trial += 1
            if t:
                time_c += 1
                time += t

            if file_req_bin not in res.keys():
                res[file_req_bin] = 0

            """ Majority decisions """
            # 1
            print('Majority decision based on g = 1:', file=ft)
            file_comp_bin = decision_maker(res, R, counter1, 1, lc_1, file_req_bin)
            file_comp_bin_v = np.fromstring(' '.join(file_comp_bin), dtype=np.int, sep=' ')
            file_comp = bin2int(file_comp_bin_v)
            bit_errors_1 += abs(file_comp_bin_v - file_req_bin_v)

            if file_comp != file_req:
                print('Error in measurement!', file=ft)
                print('Bit errors:', abs(file_comp_bin_v - file_req_bin_v), file=ft)
                print('Number of bit errors:', hamming_weight(abs(file_comp - file_req)), file=ft)
            else:
                print('Correct!', file=ft)
                acc_1 += 1

            # 2
            print('Majority decision based on g = 2:', file=ft)
            file_comp_bin = decision_maker(res, R, counter2, 2, lc_2, file_req_bin)
            file_comp_bin_v = np.fromstring(' '.join(file_comp_bin), dtype=np.int, sep=' ')
            file_comp = bin2int(file_comp_bin_v)
            bit_errors_2 += abs(file_comp_bin_v - file_req_bin_v)

            if file_comp != file_req:
                print('Error in measurement!', file=ft)
                print('Bit errors:', abs(file_comp_bin_v - file_req_bin_v), file=ft)
                print('Number of bit errors:', hamming_weight(abs(file_comp - file_req)), file=ft)
            else:
                print('Correct!', file=ft)
                acc_2 += 1

            # 3
            print('Majority decision based on g = ' + str(R << 1) + ':', file=ft)
            file_comp_bin = decision_maker(res, 0, 0, 3, lc_3, file_req_bin)
            file_comp_bin_v = np.fromstring(' '.join(file_comp_bin), dtype=np.int, sep=' ')
            file_comp = bin2int(file_comp_bin_v)
            bit_errors_3 += abs(file_comp_bin_v - file_req_bin_v)

            if file_comp != file_req:
                print('Error in measurement!', file=ft)
                print('Bit errors:', abs(file_comp_bin_v - file_req_bin_v), file=ft)
                print('Number of bit errors:', hamming_weight(abs(file_comp - file_req)), file=ft)
            else:
                print('Correct!', file=ft)
                acc_3 += 1
        else:
            print('Fail!', file=ft)

    """ Average time taken """
    if time_c:
        print('Average time taken: ' + str(time / time_c), file=fa)
    else:
        print('No time taken.', file=fa)

    """ Averages and variances for correct bits """
    # 1
    avg_1 = np.array([dd[key] / trials for dd, key in zip(counter1, file_req_bin)])
    acc_bit_1 = avg_1 / shots
    ld_var_1 = [(lc_1[i] / shots - acc_bit_1)**2 for i in range(trials)]
    var_1 = np.sum(ld_var_1, axis=0) / trials

    # 2
    avg_2 = np.array([dd[key] / trials
                      for dd, key in zip(counter2, [file_req_bin[2*i:2*i+2]
                                                    for i in range(R)])])
    acc_bit_2 = avg_2 / shots
    ld_var_2 = [(lc_2[i] / shots - acc_bit_2)**2 for i in range(trials)]
    var_2 = np.sum(ld_var_2, axis=0) / trials

    # 3
    avg_3 = np.dot(np.ones(trials), [lc_3[i] for i in range(trials)]) / trials
    acc_bit_3 = avg_3 / shots
    ld_var_3 = [(lc_3[i] / shots - acc_bit_3)**2 for i in range(trials)]
    var_3 = np.sum(ld_var_3, axis=0) / trials

    """ Majority decisions """
    # 1
    print('\nMajority decision based on g = 1:', file=fa)
    print('Accuracy for the file requested: ' + str(acc_1 / trials * 100) + ' %', file=fa)
    print('Accuracy per bit:', acc_bit_1, file=fa)
    print('Variance per bit:', var_1, file=fa)
    be_1 = bit_errors_1.dot(np.ones(R << 1, dtype=np.int))
    print('Average number of bit errors: ' + str(be_1 / trials), file=fa)
    print('Number of errors per bit in total: ' + str(bit_errors_1), file=fa)
    print('Counters for bits:', counter1, file=fa)

    # 2
    print('\nMajority decision based on g = 2:', file=fa)
    print('Accuracy for the file requested: ' + str(acc_2 / trials * 100) + ' %', file=fa)
    print('Accuracy per pairs:', acc_bit_2, file=fa)
    print('Variance per pairs:', var_2, file=fa)
    be_2 = bit_errors_2.dot(np.ones(R << 1, dtype=np.int))
    print('Average number of bit errors: ' + str(be_2 / trials), file=fa)
    print('Number of errors per bit in total: ' + str(bit_errors_2), file=fa)
    print('Counters for pairs of bits:', counter2, file=fa)

    # 3
    print('\nMajority decision based on g = ' + str(R << 1) + ':', file=fa)
    print('Accuracy for the file requested: ' + str(acc_3 / trials * 100) + ' %', file=fa)
    print('Accuracy for the measured file:', acc_bit_3, file=fa)
    print('Variance for the measured file:', var_3, file=fa)
    be_3 = bit_errors_3.dot(np.ones(R << 1, dtype=np.int))
    print('Average number of bit errors: ' + str(be_3 / trials), file=fa)
    print('Number of errors per bit in total: ' + str(bit_errors_3), file=fa)

    ft.close()
    fa.close()

def main():
    """
    Input:
        file = 0 or (1 << bit_n) - 1, file we want to request
        bit_n = number of bits for the number
        trials = number of trials for computing the averages
        device = device we want to use for the computation
    """
    file = 0
    file = (1 << 4) - 1
    bit_n = 14
    trials = 100
    device = provider.get_backend('ibmq_16_melbourne')
    quantum_experiment(file, bit_n, trials, device)


if __name__ == "__main__":
    main()
