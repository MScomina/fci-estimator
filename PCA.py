import numpy as np
import random
import dataset_generator

def PCA(dataset : np.ndarray, n_components : int | None = None, spectrum_percentage : float | None = 0.95):

    normalized_dataset = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    covariance_dataset = np.cov(normalized_dataset, ddof=1, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_dataset)

    total_variance = sum(eigenvalues)

    if n_components is None:
        current_variance = 0
        current_index = 0
        while current_variance <= spectrum_percentage*total_variance:
            current_variance += eigenvalues[-(current_index+1)]
            current_index += 1
        n_components = current_index

    reduced_space = eigenvectors[::-1][:n_components]
    reduced_space = np.transpose(reduced_space)

    reduced_eigenvalues = eigenvalues[::-1][:n_components]

    Z = normalized_dataset.dot(reduced_space)

    return Z, reduced_eigenvalues, reduced_space