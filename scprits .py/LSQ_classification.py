import numpy as np
import matplotlib.pyplot as plt


class LSQ_classification:
    '''
    Least square classification
    of points as detailed in Bishop, ch.4.1.

    Attributes: 
        W: matrix of the linear classificator
    '''
    def __init__(self):
        self.W = []

    def compute_W(self, X, tags, K):
        '''Computes the matrix of coefficients of the K 
        affine forms used for the 

        Args:
            X: list of training points [[x0, y0], ...]
            tags: list of classes to which the data points belong
                  [k0, k1, ...]
            K: number of different classes

        Returns:
            W: matrix of the linear classificator'''
        
        X = np.asarray(X)
        N = X.shape[1]
        X_tilde = np.vstack((np.ones(N), X))
        
        T = np.zeros((N, K))
        for row, tag in enumerate(tags):
            T[row, tag] = 1
            
        self.W = np.linalg.lstsq(X_tilde.transpose(), T)[0]
        
        
    
    def classifier(self, points):
        '''Uses the matrix W in order to classify
        the given points according to the largest coordinate
        of the associated linear mapping.
        
        Args:
            points: (num_pts, D) array of points.

        Returns: the array of estimated tags.
        '''
        
        if self.W == []:
            print ('First train the classifier!')
            return
        
        points = np.asarray(points)
        num_pts = points.shape[1]
        pts_tilde = np.vstack((np.ones(num_pts), points))
        
        Y = (self.W.transpose()).dot(pts_tilde)
        
        return np.argmax(Y, axis=0)
        

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
        
    K = 4
    X, tags = create_2d_data(K, sigma = 5)
    
    cls = LSQ_classification()
    cls.compute_W(X, tags, K)    
    
    
    a = 50
    grid_dim = 1000
    x = np.linspace(-a, a, grid_dim)    
    y = np.linspace(-a, a, grid_dim)    
    XX, YY = np.meshgrid(x, y)
    
    ptos = np.vstack((XX.flatten(), YY.flatten()))
    z = cls.classifier(ptos)
    ZZ = z.reshape(grid_dim, grid_dim)

    plt.plot(X[0, :], X[1, :], 'x')
    plt.contour(XX, YY, ZZ)    
    plt.show()
    

    
    

    
    
    
