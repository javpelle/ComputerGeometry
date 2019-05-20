import numpy as np
from random import sample
import matplotlib.pyplot as plt

class KMeans:
    '''
    KMeans Algorithm
    
    ----------
    Attributes
    ----------
    - k: Integer Number of Clusters. Default value is 2.
    - norm: Function. Vector Space Norm. Default
        value is euclidean norm (numpy.linalg.norm).
    '''
    
    def __init__(self, k = 2, norm = np.linalg.norm):
        self.k = k
        self.norm = norm
        
        
    def cluster(self, data, maxIter = 15, tolerance = 0.01, silent = True):
        '''
        Executes KMeans Algorithm.
        ----------
        Paramaters
        ----------
        - data: Data Matrix that will be clusterized.
        - maxIter (optional): Integer. Maximum iterations that the
            algorithm will be executed. Default value is 15.
        - tolerance (optional): Float. Default value = 0.01.
        - silent(optional): Boolean. Show log if False. Default value True.
            
        ------
        Return
        ------
        Tupla clusters, centroids, y_pred. Clusters is a list (size k) and each
        list member is a data Matrix of points. Centroids is a list (size k)
        with last centroids calculated. y_pred is a list of size len(data) with
        correspondient class to each point.
        '''
        
        data = np.array(data)
        centroids = []
        n = len(data) #Size of date
        m = len(data[0]) #Dimensions
        # Inicialice centroids randomly
        randomIndex = sample(range(0, n), self.k)
        for i in range(0, self.k):
            centroids.append(data[randomIndex[i]])
        
        isOptimal = False;
        iterations = 0;
        
        while (not isOptimal and iterations < maxIter):
            
            # Inicialize k  empty clusters
            clusters = [[] for i in range(0, self.k)]
            y_pred = []
            
            iterations += 1
                
            # Classify each individual in the cluster which minimices distance to its centroid.
            for x in data:
                distances = [self.norm(x - i) for i in centroids]
                minCluster = distances.index(min(distances))
                clusters[minCluster].append(x)
                y_pred.append(minCluster + 1)
            clusters = [np.array(i) for i in clusters]
    
            auxCentroids = list(centroids)
    
            #New centroids are the mean of each cluster
            for i in range (0, self.k):
                centroids[i] = [np.average([clusters[i][:,j]]) for j in range(0, m)]
    
    
            if max([self.norm(np.array(centroids[i]) - np.array(auxCentroids[i]))
                    /self.norm(auxCentroids[i]) for i in range (0, self.k)]) <= tolerance:
                isOptimal = True
    
                    #If at least a centroid change substantially, then we keep iterating
        if (not silent):
            print('number of iterations:', iterations)
        return clusters, centroids, y_pred



def clusterPlot(clusters, x, y, centroids):
    '''
    Draw all members of a cluster list and centroids
    
    ----------
    Paramaters
    ----------
    
    - clusters: Clusters list and each cluster is a data Matrix of points
    - x: coordinate x in graphic.
    - y: coordinate y in graphic.
    - centroids: List of points.
    '''
    for i in range(0, len(centroids)):
        cluster = clusters[i]
        p = plt.plot(cluster[:,x], cluster[:,y], 'o')
        centroid= centroids[i]
        plt.plot(centroid[x], centroid[y], 'x', markersize = 40, color = p[-1].get_color())
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

    return X, tags
    
if __name__ == '__main__':
        
    K = 10
    X, _ = create_2d_data(K, sigma = 5)
    X = X.transpose()
    
    model = KMeans(K)
    clusters, centroids, y_pred = model.cluster(X, tolerance = 0, silent = False)
    clusterPlot(clusters,0,1, centroids)