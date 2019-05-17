import pickle
from FTL_algorithms import *
import numpy as np


def load_reduced_mnist():
    (X, Y) = pickle.load(open('reduced_mnist.p', 'rb'))
    digits = {}
    for i in range(10):
        digits[i] = (X[np.argwhere(Y == i)].squeeze()+7)/17 #normalize to unit square

    return digits

def load_reduced_mnist_flat():
    (X, Y) = pickle.load(open('reduced_mnist.p', 'rb'))
    digits = {}
    for i in range(10):
        digits[i] = (X[np.argwhere(Y == i)].squeeze()+7)/17 #normalize to unit square

    return (X.squeeze(),Y)


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

    def round_robin(digits_dict):
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

    def iterate_lex():
        X,Y = load_reduced_mnist_flat()
        sorted = X[np.lexsort((X[:,1],X[:,0]))]
        for i in range(sorted.shape[0]):
            yield sorted[i,:]

    def iterate_equally_spaced_1d_lex(spacing, samples_per_site):
        X = np.linspace(0,1,spacing)
        for site in range(spacing):
            for i in range(samples_per_site):
                yield X[site]

    def iterate_equally_spaced_1d_round_robin(spacing):
        X = np.linspace(0,1,spacing)
        while True:
            for site in range(spacing):
                print('site', site)
                yield X[site]





    print(" running IC")
    iteratorIC = iter(iterate_clusters(digits_dict, T//10))
    #IC = run_ftl(lambda: next(iteratorIC), k, d, T, name='iter_clustures')

    print(" running RR")
    iteratorRR = iter(round_robin(digits_dict))
    #RR = run_ftl(lambda: next(iteratorRR), k, d, T, name='roundrobin')

    print(" running RRext")
    concat_mnist = [x for x in round_robin(digits_dict)]*10
    iteratorRRext = iter(concat_mnist)
    #RRext = run_ftl(lambda: next(iteratorRRext), k, d, T*10, name='roundrobin_ext10')

    print(" running lex")
    iteratorLex = iter(iterate_lex())
    #Lex = run_ftl(lambda: next(iteratorLex), k, d, T, name='mnist_lex')

    print(" running 1d lex")
    iteratorLex1d = iter(iterate_equally_spaced_1d_lex(int(T**(1/2))+1,int(T**(1/2))+1))
    Lex1d = run_ftl(lambda: next(iteratorLex1d), 3, 1, T, name='eq1d_lexk2')

    print(" running 1d RR")
    iteratorRR1d = iter(iterate_equally_spaced_1d_round_robin(int(T**(1/2))))
    RR1d = run_ftl(lambda: next(iteratorRR1d), 3, 1, T, name='eq1d_RRk2')




