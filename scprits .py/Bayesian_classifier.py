import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BayesianClassifier:

    '''
    Bayesian Clasifier
    
    ----------
    Attributes
    ----------
    
    - classes_set: list of tags.
    - meanCluster: list which contains the mean of each cluster.
    - inverse: list which contains the inverse of the covariance matrix of
        each cluster.
    - compute: auxiliar variable. Contains Log||covariance matrix|| -
        2*log(N_k/N) for each cluster.
    '''
    
    def fit(self, data_X, data_y):
        '''
        Fit function. 
        ----------
        Paramaters
        ----------
        - data_X: Data Matrix training set.
        - data_y: Vector of classes. data_X[i,:] class is
            data_y[i].
        '''
        
        # Prepare data. We need data in clusters
        self.classes_set = list(set(data_y))
        clusters = [[] for i in self.classes_set]
        
        # If i-data has class j, introduces it in cluster j
        for i in range(0, len(data_y)):
            clusters[self.classes_set.index(data_y[i])].append(data_X[i])
        
        k = len(clusters)
        clusters = np.array([np.array(c) for c in clusters])
        
        N = len(data_X)
        d = len(data_X[0]) #Dimensions
        
        self.meanCluster = []
        covarianceCluster = []
        sizeCluster = [len(clusters[i]) for i in range (0, k)]
        
        for i in range (0, k):

            #Compute mean of cluster
            self.meanCluster.append([np.average([clusters[i][:,j]]) for j in range(0, d)])

            #Compute the stimated covariance matrix of cluster
            aux = np.zeros((d,d))
            for j in range (0, sizeCluster[i]):
                nDisper = np.subtract(clusters[i][j], self.meanCluster[i])
                product = np.outer(nDisper,nDisper)
                aux = aux + product
            aux = aux / sizeCluster[i]
            covarianceCluster.append(aux)
            
        self.inverse = [np.linalg.inv(i) for i in covarianceCluster]
        self.compute = [np.log(np.linalg.norm(covarianceCluster[i]) -2*np.log(sizeCluster[i]/N)) for i in range(0, k)]
        
        
    def predict(self, test_X):
        '''
        Predict function.
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
        test_y = []
        
        for x in test_X:
            solutions = []
            #Compute the goal function for the cluster. 
            #We will compute the function in multiple steps

            #First we find the distance between point x and cluster's mean
            distances = [np.subtract(x, i) for i in self.meanCluster]
            
            solutions = [np.dot(np.dot(distances[i], self.inverse[i]), distances[i]) for i in range(0, len(distances))]
           
            solutions = [solutions[i] + self.compute[i] for i in range(0, len(solutions))]

            test_y.append(self.classes_set[np.argmin(solutions)])
        return np.array(test_y)
		
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

    return X, tags
	
if __name__ == '__main__':
        
    K = 5
    X, y  = create_2d_data(K, sigma = 5)
    X = X.transpose()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5)
    
    model = BayesianClassifier()
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