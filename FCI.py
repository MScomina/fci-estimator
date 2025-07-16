import numpy as np
from sympy import gamma, Float
import scipy as scy

def center_and_normalize(dataset : np.ndarray) -> np.ndarray:
    '''
        Center and normalize the dataset of shape (samples, features).
    '''

    centered_data = dataset - np.mean(dataset, axis=0)
    normalized_data = centered_data / np.std(centered_data, axis=0)
    
    return normalized_data


def analytical_FCI(x,d,x0=1):
    """
    Compute the analytical average full correlation integral on a **d**-dimensional sphere at **x**

    :param x: a real number in (0,2), or a vector of real numbers in (0,2)
    :param d: a real positive number
    :param x0: a real number (should be close to 1). It's such that f(x0)=0.5
    :returns: a real number, or a numpy vector of real numbers
    """
    return  0.5 * ( 1 + float(Float((gamma((1+d)/2)) / (np.sqrt(np.pi) * gamma(d/2) ))) * (-2+(x/x0)**2) * scy.special.hyp2f1( 0.5, 1-d/2, 3/2, 1/4 * (-2+(x/x0)**2)**2 ) )

def FCI(dataset : np.ndarray) -> np.ndarray:
    '''
        Full Correlation Integral for a dataset of shape (samples, features).
    '''
    n = dataset.shape[0]
    m = int(n*(n-1)/2)
    rs = np.empty(m)
    for i in range(n):
        for j in range(i+1,n):
            c = int( -0.5 * i *  (1 + i - 2 * n) + (j - i) - 1 )
            rs[c] = np.linalg.norm(dataset[i]-dataset[j]) 
    rs = np.sort(rs)
    r = np.empty((m,2))
    for i in range(m):
        r[i] = np.array([ rs[i] , i*1./m ])
    return r


print(analytical_FCI(np.array([k/100 for k in range(200)]),2))