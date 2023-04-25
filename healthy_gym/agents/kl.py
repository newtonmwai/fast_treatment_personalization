import numpy as np

PRECISION = 1e-8


def gaussian(x, y, sigma):
    """
    Compute the KL-divergence between univariate Gaussians with stddev sigma:

    Args:
        x: mean
        y: mean
        sigma: stddev
    """
    assert sigma > 0
    return (x - y) ** 2 / (2 * sigma ** 2)
