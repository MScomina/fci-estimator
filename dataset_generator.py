import numpy as np
import random

def generate_gaussian_dataset(dimension : int, intrinsic : int, samples : int = 300, noise : float = 0.01, influence : float = 0.3) -> np.ndarray:

    dataset = np.empty((dimension, samples))

    for k in range(intrinsic):
        dataset[k] = np.random.normal(size=samples)

    for k in range(intrinsic, dimension):
        sample = random.sample(range(intrinsic), random.randint(1,intrinsic))
        dataset[k] = np.zeros(samples)
        for j in sample:
            dataset[k] += (influence+random.random()*influence/5.0)*dataset[j]
        dataset[k] += noise*np.random.normal(size=samples)

    dataset = np.transpose(dataset)

    return dataset