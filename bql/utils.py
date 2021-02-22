import numpy as np
from numpy.random import normal, gamma
from pynverse import inversefunc
from scipy.special import digamma

def bql_f_inv(x):
    """
        Returns the inverse of f at x where:
        f(x) = log(x) - digamma(x)
    """

    # Function to take the inverse of
    def bql_f(x_):
        return np.log(x_) - digamma(x_)

    result = inversefunc(bql_f,
                         y_values=x,
                         domain=[1e-12, 1e12],
                         open_domain=True,
                         image=[1e-16, 1e16])

    return float(result)


def normal_gamma(mu0, lamda, alpha, beta):
    """
        Returns samples from Normal-Gamma with the specified parameters.

        Number of samples returned is the length of mu0, lambda, alpha, beta.
    """

    # Check if parameters are scalars or vetors
    if type(mu0) == float:
        size = (1,)
    else:
        size = mu0.shape

    # Draw samples from gamma (numpy "scale" is reciprocal of beta)
    taus = gamma(shape=alpha, scale=beta ** -1, size=size)

    # Draw samples from normal condtioned on the sampled precision
    mus = normal(loc=mu0, scale=(lamda * taus) ** -0.5, size=size)

    return mus, taus
