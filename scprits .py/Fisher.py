import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Fisher:
    '''
    Least Square classification algorithm class.
    ----------
    Attributes
    ----------
    - w: Linear vector associated to linear application.
    - c: Real. if w(x) < c then x belongs to C1, otherwise, to C2
    - classes_set: list of tags.
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
        if len(self.classes_set) != 2:
            print ('Must be 2 classes')
            return
        d = len(data_X[0])
        clusters = [[] for i in self.classes_set]
        
        
        # If i-data has class j, introduces it in cluster j
        for i in range(0, len(data_y)):
            clusters[self.classes_set.index(data_y[i])].append(data_X[i])
        
        clusters = np.array([np.array(c) for c in clusters])
        
        k1 = len(clusters[0])
        k2 = len(clusters[1])
        N = len(data_X)
        
        mean1 = np.sum(clusters[0], axis = 0) / k1
        mean2 = np.sum(clusters[1], axis = 0) / k2
        
        SW1 = np.zeros((d,d))
        SW2 = np.zeros((d,d))
        
        for x in clusters[0]:
            aux = np.subtract(x, mean1)
            SW1 = SW1 + np.outer(aux, aux)
            
        for x in clusters[1]:
            aux = np.subtract(x, mean2)
            SW2 = SW2 + np.outer(aux, aux)
            
        SW = SW1 + SW2
        self.w = np.linalg.inv(SW).dot(np.subtract(mean2, mean1))
        
        m1 = self.w.dot(mean1)
        m2 = self.w.dot(mean1)
        
        p1 = k1 / N
        p2 = k2 / N
        
        sigma1 = 0
        sigma2 = 0
        
        for x in clusters[0]:
            sigma1 = sigma1 + (self.w.dot(x) - m1) ** 2
        sigma1 = sigma1 / k1
            
        for x in clusters[1]:
            sigma2 = sigma2 + (self.w.dot(x) - m2) ** 2
        sigma2 = sigma2 / k2
        
        # Computes F(c) = 0 roots. They are our relative extremas
        a2 = (1/(2*sigma1) - 1/(2*sigma2))
        a1 = (m1/sigma1 - m2/sigma2)
        a0 = np.log(p1 / np.sqrt(sigma1)) - np.log(p2 / np.sqrt(sigma2)) + (m2**2)/(2*sigma2) - (m1**2)/(2*sigma1)
        c_list = np.roots([a2, a1, a0])
        
        if self.F_derivate(c_list[0], m1, m2, sigma1, sigma2) > 0:
            # If F'(c1) > 0 then c1 is our minimun
            self.c = c_list[0]
        else:
            # If F'(c2) > 0 then c2 is our minimun
            self.c = c_list[1]
    
    def F_derivate(self, c, m1, m2, sigma1, sigma2):
        '''
        Auxiliar F'(c) function.
        
        ------
        Return
        ------
        - F'(c)
        '''
        return (m1 - c)/sigma1 + (c - m2)/sigma2
        
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
        aux = self.w.dot(test_X.T)
        
        test_y = [self.classes_set[0] if i < self.c else self.classes_set[1] for i in aux]
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

    return X.T, tags
    
if __name__ == '__main__':
        
    K = 2
    X, y  = create_2d_data(K, sigma = 6)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5)
    
    model = Fisher()
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
    
    a = 50
    grid_dim = 1000
    x = np.linspace(-a, a, grid_dim)    
    y = np.linspace(-a, a, grid_dim)    
    XX, YY = np.meshgrid(x, y)
    
    ptos = np.vstack((XX.flatten(), YY.flatten()))
    z = model.predict(ptos.T)
    ZZ = z.reshape(grid_dim, grid_dim)
    plt.plot(train_X[:, 0], train_X[:, 1], 'x')
    plt.contour(XX, YY, ZZ)    
    plt.show()