import numpy as np
from sklearn.linear_model import LinearRegression
import random
import dataset_generator

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def two_nn(dataset):
    n_points = dataset.shape[0]
    mu_values = np.zeros(n_points)
    N = len(dataset)

    for i in range(n_points):
        distances = []
        
        for j in range(n_points):
            if i != j:
                dist = euclidean_distance(dataset[i], dataset[j])
                distances.append((dist, j))
        
        distances.sort(key=lambda x: x[0])

        r1 = distances[0][0]
        r2 = distances[1][0]
        
        mu = r2 / r1
        mu_values[i] = mu

    mu_values = mu_values[np.argsort(mu_values)]

    Femp = np.arange(int(N)) / N

    Ir = LinearRegression(fit_intercept=False)
    Ir.fit(np.log(mu_values).reshape(-1,1), -np.log(1 - Femp).reshape(-1,1))

    d = Ir.coef_[0][0]

    return d