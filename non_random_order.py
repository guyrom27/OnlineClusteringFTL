import pickle
from FTL_algorithms import *
import numpy as np


def load_reduced_mnist():
    (X, Y) = pickle.load(open('reduced_mnist.p', 'rb'))
    digits = {}
    for i in range(10):
        digits[i] = (X[np.argwhere(Y == i)].squeeze()+7)/17 #normalize to unit square

    return digits


def run_ftl(gen_data, k, d, T, name):
    ftl = FTL_online(k, d)
    ftl_res = run(ftl,T, gen_data)
    pickle.dump(ftl_res[0], open(name+".ftl.p", "wb"))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ftl_res[0], label='ftl')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('regret')
    ax.set_title(name + ' regret vs. t')
    plt.savefig(name)

    return ftl_res

if __name__ == '__main__':

    digits_dict = load_reduced_mnist()

    T = 174*10
    d = 2
    k = 10

    def round_robbin(digits_dict):
        min_len = min([x.shape[0] for x in digits_dict.values()])
        for i in range(min_len):
            for dig in digits_dict:
                yield digits_dict[dig][i,:]

    def iterate_clusters(digits_dict, samples_per_cluster):
        print(samples_per_cluster)
        min_len = min([x.shape[0] for x in digits_dict.values()])
        assert(samples_per_cluster <= min_len)
        for dig in digits_dict:
            for i in range(samples_per_cluster):
                yield digits_dict[dig][i,:]


    print(" running IC")
    iteratorIC = iter(iterate_clusters(digits_dict, T//10))
    IC = run_ftl(lambda: next(iteratorIC), k, d, T, name='iter_clustures')

    print(" running RR")
    iteratorRR = iter(round_robbin(digits_dict))
    RR = run_ftl(lambda: next(iteratorRR), k, d, T, name='roundrobin')
