import numpy as np
from sympy import gamma, Float
import scipy as scy
import scipy.optimize as scyopt
import dataset_generator

"""
    Note: Some code has been adapted and inspired from the repository: https://github.com/vittorioerba/pyFCI
"""

def center_and_normalize(dataset : np.ndarray) -> np.ndarray:
    """
        Centers and normalizes the dataset's samples to have mean 0 and norm 1.
    """

    if len(dataset) != 1:
        centered_data = dataset - np.mean(dataset, axis=0)
    else:
        centered_data = dataset
    norms = np.linalg.norm(centered_data, axis=1).reshape(-1, 1)
    norms[norms==0] = 1
    normalized_data = centered_data / norms
    
    return normalized_data

def FCI(dataset : np.ndarray, samples : int | None = None) -> np.ndarray:

    """
        Computes the FCI of a given dataset.

        :param dataset: The dataset to compute the FCI of.
        :param samples: The amount of samples randomly chosen from dataset. If None, FCI will use all samples.
    """
    if samples is not None:
        if samples > len(dataset):
            samples = len(dataset)
        dataset = dataset[np.random.choice(len(dataset), size=samples, replace=False)]
    else:
        samples = len(dataset)

    n = samples
    m = int(n*(n-1)/2)
    rs = np.empty(m)
        
    diff = dataset[:, np.newaxis, :] - dataset[np.newaxis, :, :]
    squared_distances = np.sum(diff ** 2, axis=-1)
    upper_triangular_indices = np.triu_indices_from(squared_distances, k=1)
    rs = np.sort(np.sqrt(squared_distances[upper_triangular_indices]))
    r = np.empty((m,2))

    r[:, 0] = rs
    r[:, 1] = np.arange(m) / m

    return r

def analytical_FCI(x,d,x0=1):
    """
    Compute the analytical average full correlation integral on a **d**-dimensional sphere at **x**

    :param x: a real number in (0,2), or a vector of real numbers in (0,2)
    :param d: a real positive number
    :param x0: a real number (should be close to 1). It's such that f(x0)=0.5
    :returns: a real number, or a numpy vector of real numbers
    """
    return 0.5 * (1 + float(Float((gamma((1+d)/2)) / (np.sqrt(np.pi) * gamma(d/2)))) * (-2+(x/x0)**2) * scy.special.hyp2f1(0.5, 1-d/2, 3/2, 1/4 * (-2+(x/x0)**2)**2))

def fit_FCI(rho, samples=500, threshold=0.1):
    """
        Given an empirical full correlation integral **rho**, it tries to fit it to the analytical_FCI curve.
        To avoid slow-downs, only a random sample of **samples** points is used in the fitting.
        If the fit fails, it outputs [0,0,0]

        :param rho: vector of shape (N,2) of points in (0,2)x(0,1)
        :param samples: a positive integer
        :returns: the fitted dimension, the fitted x0 parameter and the mean square error between the fitted curve and the empirical points
    """
    samples = min(len(rho),samples)
    data = rho[np.random.choice(len(rho),samples)]

    fit = scyopt.curve_fit(analytical_FCI, data[:,0], data[:,1])
    if abs(fit[0][1] - 1)>threshold:
        return [0,0,0]
    else:
        mse = np.sqrt(np.mean([ (pt[1] - analytical_FCI(pt[0],fit[0][0],fit[0][1]))**2 for pt in data ]))
        return [fit[0][0]+1,fit[0][1],mse]

def local_FCI(dataset : np.ndarray, center : int, neighbours : int = 100, radius : float | None = None):
    """
        Computes the local FCI around dataset[center].
        :param dataset: The dataset to compute the FCI of.
        :param center: The index of the center picked for the local FCI.
        :param neighbours: The amount of neighbours to pick around the center. Defaults to 100.
        :param radius: The radius of the ball to include points from **in normalized space**. Overwrites **neighbours**.
    """

    center_coor = dataset[center]
    distances = np.linalg.norm(dataset-center_coor, axis=1)
    if radius is not None:
        dataset = center_and_normalize(dataset)
        center_coor = dataset[center]
        distances = np.linalg.norm(dataset-center_coor, axis=1)
        mask = (distances<=radius) & (distances>0.0001)
        dataset = dataset[mask]
    else:
        closest_points_idx = np.argsort(distances)
        closest_points_idx = np.delete(closest_points_idx, 0)
        dataset = dataset[closest_points_idx[:neighbours]]
    # Averaged dataset is not guaranteed anymore, have to re-center.
    dataset = center_and_normalize(dataset)
    if len(dataset) <= 2:
        # Too few points found, return d = 0
        return [0,0,0]
    else:
        return fit_FCI(FCI(dataset))

dataset = dataset_generator.generate_gaussian_dataset(1000,50,samples=1000,influence=0.0, noise=0.0)
fci = local_FCI(dataset,0,neighbours=20)
print(fci[0])