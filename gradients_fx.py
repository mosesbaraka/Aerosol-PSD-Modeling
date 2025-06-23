import numpy as np
from scipy.special import erf

# Helper function for u_k(x)
def u_k(x, h, mu_k, sigma_k):
    return (np.log(x - h) - mu_k) / sigma_k

# Helper function for the normalization constant
def compute_C(x_0, x_n, mu_k, sigma_k, h, lambda_k):
    C = 1 / np.sum(
        [lambda_k[i] * (erf((np.log(x_n - h) - mu_k[i]) / (np.sqrt(2) * sigma_k[i])) -
                        erf((np.log(x_0 - h) - mu_k[i]) / (np.sqrt(2) * sigma_k[i])))
         for i in range(2)]
    )
    return C

def derivative_wrt_x(x, x_0, x_n, mu_k, sigma_k, h, lambda_k):
    C = compute_C(x_0, x_n, mu_k, sigma_k, h, lambda_k)
    derivative = 0
    for i in range(2):
        u_i = u_k(x, h, mu_k[i], sigma_k[i])
        exp_term = np.exp(-u_i**2 / 2)
        derivative += lambda_k[i] * (exp_term / ((x - h) * sigma_k[i] * np.sqrt(2 * np.pi)))
    return C * derivative

def derivative_wrt_mu_k(x, x_0, x_n, mu_k, sigma_k, h, lambda_k):
    """
    Computes the derivative of n^\circ with respect to mu_k for both components k = 1, 2.
    Returns a list with derivatives for k=1 and k=2.
    """
    C = compute_C(x_0, x_n, mu_k, sigma_k, h, lambda_k)
    derivatives = []
    for i in range(2):  # k=1, 2
        u_i = u_k(x, h, mu_k[i], sigma_k[i])
        exp_term = np.exp(-u_i**2 / 2)
        derivative = lambda_k[i] * (-exp_term / (sigma_k[i] * np.sqrt(2 * np.pi)))
        derivatives.append(C * derivative)
    return derivatives  # Two outputs, one for each k

def derivative_wrt_sigma_k(x, x_0, x_n, mu_k, sigma_k, h, lambda_k):
    """
    Computes the derivative of n^\circ with respect to sigma_k for both components k = 1, 2.
    Returns a list with derivatives for k=1 and k=2.
    """
    C = compute_C(x_0, x_n, mu_k, sigma_k, h, lambda_k)
    derivatives = []
    for i in range(2):  # k=1, 2
        u_i = u_k(x, h, mu_k[i], sigma_k[i])
        exp_term = np.exp(-u_i**2 / 2)
        ln_term = np.log(x - h) - mu_k[i]
        derivative = lambda_k[i] * (-ln_term * exp_term / (sigma_k[i]**2 * np.sqrt(2 * np.pi)))
        derivatives.append(C * derivative)
    return derivatives  # Two outputs, one for each k


def derivative_wrt_h(x, x_0, x_n, mu_k, sigma_k, h, lambda_k):
    C = compute_C(x_0, x_n, mu_k, sigma_k, h, lambda_k)
    derivative = 0
    for i in range(2):
        u_i = u_k(x, h, mu_k[i], sigma_k[i])
        exp_term = np.exp(-u_i**2 / 2)
        term = exp_term / ((x - h) * sigma_k[i] * np.sqrt(2 * np.pi))
        derivative -= lambda_k[i] * term
    return C * derivative

