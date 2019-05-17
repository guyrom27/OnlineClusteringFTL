import numpy as np
from sklearn.cluster import KMeans
from math import log
import random




class FTL_online:
    def __init__(self, k, d):
        self.current = FTL(np.array([[0] * d for i in range(k)]), k, d)
        self.next = self.current
        self.X = []
        self.k = d
        self.d = d

    def choose(self):
        self.current = self.next

    def observeOutcome(self, x_t):
        self.X.append(x_t)
        self.next = FTL(np.array(self.X), self.k, self.d)

    def loss(self, x_t):
        return self.current.loss(x_t)

    def get_prophet_loss_so_far(self):
        return self.next.kmeans.inertia_ /self.d


class FTL:
    def __init__(self, X, k, d):
        if (X.shape[0] < k):
            self.kmeans = KMeans(n_clusters=k).fit(np.array([[0]*d for i in range(k)]).reshape(-1, d))
        else:
            self.kmeans = KMeans(n_clusters=k).fit(X.reshape(-1, d))
        self.d = d

    def loss(self, x_t):
        return -self.kmeans.score(x_t.reshape(-1, self.d))/self.d


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
            self.chosen_expert = FTL(np.array([[0] * self.d for i in range(self.k)]), self.k, self.d)
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
        return self.FTLs[-1].kmeans.inertia_ /self.d

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
        l_t = alg.loss(x_t)
        online_loss += l_t
        X.append(x_t)
        prophet = alg.get_prophet_loss_so_far()
        regret.append(online_loss-prophet)
    print('final regret', regret[-1])
    print('online', online_loss, 'prophet',prophet)
    return (regret, online_loss, prophet, alg)