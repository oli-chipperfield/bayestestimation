import numpy as np
import scipy as scipy


def _calculate_kde(draws, num=10000):
    # Estimates a KDE distribution from the posterior draws
    kde = scipy.stats.gaussian_kde(draws)
    x = np.linspace(np.min(draws), np.max(draws), num=num)
    kde_density = kde(x)
    return x, kde_density


def _calculate_map(draws, num=10000):
    # Estimates the MAP based on the maxima of the KDE estimate
    x, kde_density = _calculate_kde(draws, num=num)
    return x[np.argmax(kde_density)]
