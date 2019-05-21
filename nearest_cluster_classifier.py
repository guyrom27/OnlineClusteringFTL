import pickle
import numpy as np
from sklearn.cluster import KMeans

T = 174*10
d = 2
k = 10


class FTL_online_classifier:
    DUMMY_X = np.array([[0] * d for i in range(k)])
    DUMMY_Y = np.array([0 for i in range(k)])
    
    def __init__(self, k, d):
        self.current = FTL_nearest_cluster_classifier(self.DUMMY_X, self.DUMMY_Y, k, d)
        self.next = self.current
        self.X = []
        self.Y = []
        self.k = d
        self.d = d

    def choose(self):
        self.current = self.next

    def observeOutcome(self, x_t,y_t):
        self.X.append(x_t)
        self.Y.append(y_t)
        self.next = FTL_nearest_cluster_classifier(np.array(self.X),np.array(self.Y), self.k, self.d)

    def classify(self, x_t):
        return self.current.classify(x_t)


class FTL_nearest_cluster_classifier:
    def __init__(self, X, Y, k, d):
        if (X.shape[0] < k):
            self.kmeans = KMeans(n_clusters=k).fit(np.array([[0]*d for i in range(k)]).reshape(-1, d))
        else:
            self.kmeans = KMeans(n_clusters=k).fit(X.reshape(-1, d))
        self.d = d
        self.X = X
        self.Y = Y

    def classify(self, x_t):
        cl_labels = self.kmeans.predict(self.X)
        new_cl_label = self.kmeans.predict(np.array([x_t]).reshape(1,-1))[0]
        
        maj = np.argmax(np.bincount(self.Y[cl_labels == new_cl_label]))
        return maj



def run(alg, T, gen_labelled_data):
    mislabels = 0
    mislabels_arr = [0]
    for t in range(1,T+1):
        if (t%50==0):
            print ('iter',t)
        alg.choose()
        (x_t,y_t) = gen_labelled_data()
        mislabels += 0 if alg.classify(x_t) == y_t else 1
        alg.observeOutcome(x_t, y_t)
        mislabels_arr.append(1-mislabels/t)
    print('final accuracy', mislabels_arr[-1])
    return (mislabels_arr,mislabels_arr[-1])    
    
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
    ftl = FTL_online_classifier(k, d)
    ftl_res = run(ftl,T, gen_data)
    pickle.dump(ftl_res[0], open(name+".ftl.p", "wb"))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ftl_res[0], label='ftl')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('accuracy')
    ax.set_title(name + ' accuracy vs. t')
    plt.savefig(name)

    return ftl_res

if __name__ == '__main__':

    digits_dict = load_reduced_mnist()

    def round_robin(digits_dict):
        min_len = min([x.shape[0] for x in digits_dict.values()])
        for i in range(min_len):
            for dig in digits_dict:
                yield (digits_dict[dig][i,:], dig)

    def iterate_clusters(digits_dict, samples_per_cluster):
        print(samples_per_cluster)
        min_len = min([x.shape[0] for x in digits_dict.values()])
        assert(samples_per_cluster <= min_len)
        for dig in digits_dict:
            for i in range(samples_per_cluster):
                yield (digits_dict[dig][i,:], dig)


    print(" running IC")
    iteratorIC = iter(iterate_clusters(digits_dict, T//10))
    IC = run_ftl(lambda: next(iteratorIC), k, d, T, name='iter_clustures_classifier')

    print(" running RR")
    iteratorRR = iter(round_robin(digits_dict))
    RR = run_ftl(lambda: next(iteratorRR), k, d, T, name='roundrobin_classifier')

    print(" running RRext")
    concat_mnist = [x for x in round_robin(digits_dict)]*10
    iteratorRRext = iter(concat_mnist)
    RRext = run_ftl(lambda: next(iteratorRRext), k, d, T*10, name='roundrobin_ext10_classifier')
