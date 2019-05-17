import numpy as np
from numpy.random import normal
from random import randint
from sklearn.cluster import KMeans

import pickle

class FTL_online:
    def __init__(self, k, d):
        self.current = FTL(np.array([[0] * d for i in range(k)]), k, d)
        self.next = self.current
        self.X = []

    def choose(self):
        self.current = self.next

    def observeOutcome(self, x_t):
        self.X.append(x_t)
        self.next = FTL(np.array(self.X), k, d)

    def loss(self, x_t):
        return self.current.loss(x_t)

    def get_prophet_loss_so_far(self):
        return self.next.kmeans.inertia_


class FTL:
    def __init__(self, X, k, d):
        if (X.shape[0] < k):
            self.kmeans = KMeans(n_clusters=k).fit(np.array([[0]*d for i in range(k)]).reshape(-1, d))
        else:
            self.kmeans = KMeans(n_clusters=k).fit(X.reshape(-1, d))
        self.d = d

    def loss(self, x_t):
        return -self.kmeans.score(x_t.reshape(-1, self.d))/self.d


def sample_k_gaussians(MUs, SIGs):
    k_ind = randint(0, len(MUs) - 1)
    (mu, sig) = MUs[k_ind], SIGs[k_ind]
    in_box = False
    while (not in_box):
        p = normal(mu, sig)
        in_box = np.all(np.logical_and(p >= 0, p <= 1))
    return p


from math import log, exp
import random


# MWUA: the multiplicative weights update algorithm
class MWUA_FTL:
    def __init__(self, T, k, d):
        self.FTLs = []
        self.learning_rate = T ** (-1 / 2) * log(T)
        self.X = []
        self.weights = []
        self.chosen_expert = None
        self.k = k
        self.d = d

    def choose(self):
        if self.chosen_expert is None:
            self.chosen_expert = FTL(np.array([[0] * d for i in range(k)]), self.k, self.d)
        else:
            self.chosen_expert = self.FTLs[self.draw()]

    def observeOutcome(self, x_t):
        new_leader = FTL(np.array(self.X + [x_t]), self.k, self.d)
        self.FTLs.append(new_leader)
        self.weights.append(1)
        for x in self.X:
            single_loss = new_leader.loss(x)
            self.weights[-1] *= (1 - self.learning_rate * single_loss)
        for i, obj in enumerate(self.FTLs):
            self.weights[i] *= (1 - self.learning_rate * obj.loss(x_t))
        self.X.append(x_t)

    def loss(self, x_t):
        if (self.chosen_expert is None):
            return 1
        return self.chosen_expert.loss(x_t)

    def get_prophet_loss_so_far(self):
        return self.FTLs[-1].kmeans.inertia_

    def draw(self):
        choice = random.uniform(0, sum(self.weights))
        choiceIndex = 0

        for weight in self.weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1

def run(alg, T, gen_data):
    online_loss = 0
    regret = []
    X = []
    for t in range(T):
        if (t%50==0):
            print ('iter',t)
        alg.choose()
        x_t = gen_data()
        alg.observeOutcome(x_t)
        l_t = alg.loss(x_t)*d #it was normalized internali
        online_loss += l_t
        X.append(x_t)
        prophet = alg.get_prophet_loss_so_far()
        regret.append(online_loss-prophet)
    print('final regret', regret[-1])
    print('online', online_loss, 'prophet',prophet)
    return (regret, online_loss, prophet, alg)


def toMUs(mus, d):
    return [np.array(mu_i) for mu_i in mus]

def toSIGs(sigmas, d):
    return [np.array([sig]*d) for sig in sigmas]

def compare(gen_data, k, d, T, name):
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

    will_3k = compare(lambda : sample_k_gaussians(*mu_sig),k,d,T, name = 'gaussians-well_separated2D3k')
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
    ill = compare(lambda : sample_k_gaussians(*mu_sig),k,d,T, name = 'gaussians-ill_separated2D3k')
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
    well_4kBut3kalg = compare(lambda : sample_k_gaussians(*mu_sig),k,d,T, name = 'gaussians-well_separated-data2D4k-using3k')
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



    k10res = compare(lambda : next(iterator),k,d,T, name = 'MNISTk10')
    pickle.dump(k10res[0], open("mwua10k.p","wb"))
    pickle.dump(k10res[1], open("ftl10k.p","wb"))
    T = 10000
    d = 784
    k = 5

    iterator = iter(gen(mnist))
    k5res = compare(lambda : next(iterator),k,d,T, name = 'MNISTk5')
    pickle.dump(k5res[0], open("mwua5k.p","wb"))
    pickle.dump(k5res[1], open("ftl5k.p","wb"))