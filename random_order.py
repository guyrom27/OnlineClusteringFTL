import pickle
from numpy.random import normal
from random import randint

from math import log, exp

from FTL_algorithms import *

def sample_k_gaussians(MUs, SIGs):
    k_ind = randint(0, len(MUs) - 1)
    (mu, sig) = MUs[k_ind], SIGs[k_ind]
    in_box = False
    while (not in_box):
        p = normal(mu, sig)
        in_box = np.all(np.logical_and(p >= 0, p <= 1))
    return p

def toMUs(mus, d):
    return [np.array(mu_i) for mu_i in mus]

def toSIGs(sigmas, d):
    return [np.array([sig]*d) for sig in sigmas]

def compare_ftl_2_ftl_mwua(gen_data, k, d, T, name):
    ftl = FTL_online(k, d)
    mwua_ftl = MWUA_FTL(T, k,d)

    mwua_res = run(mwua_ftl,T, gen_data)
    ftl_res = run(ftl,T, gen_data)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(mwua_res[0], label='ftl-mwua')
    ax.plot(ftl_res[0], label='ftl')
    ax.legend()
    plt.savefig(name)

    return mwua_res,ftl_res

if __name__ == '__main__':

    T = 10000
    d = 2
    k = 3
    mu1 = (0.3,0.3)
    mu2 = (0.4,0.7)
    mu3 = (0.7,0.3)
    sigma = 0.1

    mu_sig =(toMUs((mu1,mu2,mu3),d), toSIGs((sigma,sigma,sigma),d))

    will_3k = compare_ftl_2_ftl_mwua(lambda : sample_k_gaussians(*mu_sig),k,d,T, name = 'gaussians-well_separated2D3k')
    pickle.dump(will_3k[0], open("mwua_will_3k.p","wb"))
    pickle.dump(will_3k[1], open("ftl_will_3k.p","wb"))



    T = 10000
    d = 2
    k = 3
    mu1 = (0.43,0.43)
    mu2 = (0.5,0.5)
    mu3 = (0.43,0.5)
    sigma = 0.1
    mu_sig =(toMUs((mu1,mu2,mu3),d), toSIGs((sigma,sigma,sigma),d))
    ill = compare_ftl_2_ftl_mwua(lambda : sample_k_gaussians(*mu_sig),k,d,T, name = 'gaussians-ill_separated2D3k')
    pickle.dump(ill[0], open("mwua_ill.p","wb"))
    pickle.dump(ill[1], open("ftl_ill.p","wb"))

    T = 10000
    d = 2
    k = 3
    mu1 = (0.7,0.7)
    mu2 = (0.4,0.7)
    mu3 = (0.7,0.3)
    mu4 = (0.3,0.3)
    sigma = 0.1
    mu_sig =(toMUs((mu1,mu2,mu3,mu4),d), toSIGs((sigma,sigma,sigma,sigma),d))
    well_4kBut3kalg = compare_ftl_2_ftl_mwua(lambda : sample_k_gaussians(*mu_sig),k,d,T, name = 'gaussians-well_separated-data2D4k-using3k')
    pickle.dump(well_4kBut3kalg[0], open("mwua_well_4kBut3kalg.p","wb"))
    pickle.dump(well_4kBut3kalg[1], open("ftl_well_4kBut3kalg.p","wb"))

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home='./')
    from numpy.random import shuffle
    shuffle(mnist.data)

    T = 10000
    d = 784
    k = 10

    def gen(mnist):
        for i in range(mnist.data.shape[0]):
            yield mnist.data[i,:]/255.0

    iterator = iter(gen(mnist))



    k10res = compare_ftl_2_ftl_mwua(lambda : next(iterator),k,d,T, name = 'MNISTk10')
    pickle.dump(k10res[0], open("mwua10k.p","wb"))
    pickle.dump(k10res[1], open("ftl10k.p","wb"))
    T = 10000
    d = 784
    k = 5

    iterator = iter(gen(mnist))
    k5res = compare_ftl_2_ftl_mwua(lambda : next(iterator),k,d,T, name = 'MNISTk5')
    pickle.dump(k5res[0], open("mwua5k.p","wb"))
    pickle.dump(k5res[1], open("ftl5k.p","wb"))