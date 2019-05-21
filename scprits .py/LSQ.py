import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LSQ:
    '''
    Least Square classification algorithm class.
    ----------
    Attributes
    ----------
    - W: Linear Matrix associated to affine application.
    '''
    
    def fit(self, data_X, data_y):
        '''
        Fit function. Computes W.
        ----------
        Paramaters
        ----------
        - data_X: Data Matrix training set.
        - data_y: Vector of classes. data_X[i,:] class is
            data_y[i].
        '''
        data_X = np.array(data_X)
        print(data_y)
        K = len(set(data_y)) # Number of tags
        N = data_X.shape[0] # Number of data
        X = np.vstack((np.ones(N), data_X.T))
        A = X.dot(X.T)
        T = np.zeros((K, N))
        for i in range(0 ,N):
            T[data_y[i], i] = 1
        B = X.dot(T.T)
        # AW = B
        self.W = np.linalg.solve(A, B)
    

    def predict(self, test_X):
        '''
        Predict function, executes LSQ algorithm.
        ----------
        Paramaters
        ----------
        - test_X: Data Matrix. Function will predict the
            class for each data test_X[i,:].
        
        ------
        Return
        ------
        - test_y: Vector of classes. test_X[i,:] class is
            test_y[i].
        '''
        test_X = np.array(test_X)
        N = test_X.shape[0]
        X = np.vstack((np.ones(N), test_X.T))
        T = (self.W.T).dot(X)
        return np.argmax(T, axis=0)


def clusterPlot(clusters, x, y):
    '''
    Draw all members of a cluster list and centroids
    
    ----------
    Paramaters
    ----------
    
    - clusters: Clusters list and each cluster is a data Matrix of points
    - x: coordinate x in graphic.
    - y: coordinate y in graphic.
    '''
    for cluster in clusters:
        plt.plot(cluster[:,x], cluster[:,y], 'o')
    plt.show()


def create_2d_data(K, sigma_class=10, sigma=0.5, min_num=10, max_num=20):
    '''Creates some random 2D data for testing.
    Return points X belonging to K classes given
    by tags. 
    
    Args:
        K: number of classes
        sigma_class: measures how sparse are the classes
        sigma: measures how clustered around its mean are 
               the points of each class
        min_num, max_num: minimum and maximum number of points of each class

    Returns:
        X: (N, 2) array of points
        tags: list of tags [k0, k1, ..., kN]
    '''
    
    tags = []
    N_class_list = []
    mu_list = []

    mu_list = [np.random.randn(2)*sigma_class]
    
    for k in range(1, K):
        try_mu = np.random.randn(2)*sigma_class
        while min([np.linalg.norm(mu - try_mu) for mu in mu_list]) < 2*sigma_class:
            try_mu = np.random.randn(2)*sigma_class
        mu_list.append(try_mu)

    for k in range(K):
        N_class = np.random.randint(min_num, max_num)
        tags += [k] * N_class
        N_class_list += [N_class]

    N = sum(N_class_list)
    X = np.zeros((2, N))
    count = 0
    for k in range(K):
        X[:, count:count + N_class_list[k]] = \
            mu_list[k][:, np.newaxis] + np.random.randn(2, N_class_list[k])*sigma
        count += N_class_list[k]

    return X.T, tags
    
if __name__ == '__main__':
        
    K = 3
    X, y  = create_2d_data(K, sigma = 6)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5)
    
    model = LSQ()
    model.fit(train_X, train_y)
    
    pred_y = model.predict(test_X)
    
    
    classes_set = list(set(pred_y))
    clusters = [[] for i in classes_set]
    # If i-data has class j, introduces it in cluster j
    for i in range(0, len(pred_y)):
        clusters[classes_set.index(pred_y[i])].append(test_X[i])
                
    clusters = [np.array(c) for c in clusters]
    clusterPlot(clusters,0,1)
    
    classes_set = list(set(test_y))
    clusters = [[] for i in classes_set]
    # If i-data has class j, introduces it in cluster j
    for i in range(0, len(test_y)):
        clusters[classes_set.index(test_y[i])].append(test_X[i])
                
    clusters = [np.array(c) for c in clusters]
    clusterPlot(clusters,0,1)
    
    print('Accuracy: ', np.round(np.mean(pred_y == test_y)*100, 2),'%')