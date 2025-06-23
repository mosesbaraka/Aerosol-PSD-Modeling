import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.stats import norm
from math import comb
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Function for scaled log transform
def u(x, mu, sigma, h):
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    if np.any(x <= h):  # Check if any value in x violates the condition
        raise ValueError(f"Some values in x are invalid for the given h = {h}. Logarithm is undefined.")
    return (np.log(x - h) - mu) / sigma

# Log-normal transformation derivative
def u_prime(x, sigma, h):
    """
    Compute the derivative of the log-normal transformation.
    """
    return 1 / (sigma * (x - h))

def varphi(z):
    """
    Standard normal probability density function.
    """
    return 1 / (np.sqrt(2 * np.pi)) * np.exp(-z**2 / 2)

# C function based on mixture of log-normals
def norm_C(x0, xm, mu1, sigma1, mu2, sigma2, h, lambda1=0.49, lambda2=0.51):
    """
    Compute the normalization constant C for a mixture of two log-normal distributions.
    """
    u_min = u(x0, mu1, sigma1, h)
    u_max = u(xm, mu1, sigma1, h)
    v_min = u(x0, mu2, sigma2, h)
    v_max = u(xm, mu2, sigma2, h)

    denom_part1 = lambda1 * (erf(u_max / np.sqrt(2)) - erf(u_min / np.sqrt(2)))
    denom_part2 = lambda2 * (erf(v_max / np.sqrt(2)) - erf(v_min / np.sqrt(2)))
    
    kst = 0.5 * denom_part1 + 0.5 * denom_part2
    return 1 / kst

# E(x^r)
def compute_moments(lambda_k, mu_k, sigma_k, h, x_0, x_n, C=1):
    """
    Compute the first to fourth moments for a mixture of two log-normal distributions.
    """
    moments = []
    
    # Pre-compute u(x_0) and u(x_n) for each k
    u_x_0 = [u(x_0, mu_k[k], sigma_k[k], h) for k in range(2)]
    u_x_n = [u(x_n, mu_k[k], sigma_k[k], h) for k in range(2)]

    # Iterate over r = {0, 1, 2, 3, 4}
    for r in range(5):
        moment_r = 0

        # Define the maximum allowed value for the exponential term
        max_value = 1e10
        
        # Compute moments
        for j in range(r + 1):
            binomial_coeff = comb(r, j)
            erf_terms = [
                erf((u_x_n[k] - j * sigma_k[k]) / np.sqrt(2)) - erf((u_x_0[k] - j * sigma_k[k]) / np.sqrt(2))
                for k in range(2)
            ]
            
            term = 0
            for k in range(2):
                log_term = j * mu_k[k] + 0.5 * (j * sigma_k[k])**2
                if log_term > np.log(max_value):  # Check if the logarithm exceeds the cap
                    exp_term = max_value
                else:
                    exp_term = np.exp(log_term)
                
                term += (binomial_coeff * lambda_k[k] * h**(r - j) * exp_term / 2 * erf_terms[k])
            
            moment_r += term
        
        # # Sum over j = 0 to r
        # for j in range(r + 1):
        #     binomial_coeff = comb(r, j)
            
        #     # Calculate erf terms for each k
        #     erf_terms = [
        #         erf((u_x_n[k] - j * sigma_k[k]) / np.sqrt(2)) - erf((u_x_0[k] - j * sigma_k[k]) / np.sqrt(2))
        #         for k in range(2)
        #     ]
            
        #     # Compute the moment term
        #     term = 0
        #     for k in range(2):
        #         term += (binomial_coeff * lambda_k[k] * h**(r - j) *
        #                  np.exp(j * mu_k[k] + 0.5 * (j * sigma_k[k])**2) / 2 * erf_terms[k])
            
        #     # Add the term to the moment
        #     moment_r += term
        
        # Multiply by the constant C
        moments.append(moment_r * C)
    
    # Create a DataFrame for the moments
    moments_df = pd.DataFrame(moments, columns=["E[X^r]"], index=[f"r={r}" for r in range(5)])
    
    return moments_df

# Statistics calculations
def compute_statistics(lambda_k, mu_k, sigma_k, h, x_0, x_n, C=1):
    """
    Compute the statistics: mean, variance, standard deviation, skewness, and kurtosis.
    """
    moments_df = compute_moments(lambda_k, mu_k, sigma_k, h, x_0, x_n, C)
    
    # Mean: first moment
    mean = moments_df["E[X^r]"].iloc[1]
    
    # Variance: second moment - first moment^2
    variance = moments_df["E[X^r]"].iloc[2] - mean**2
    
    # Standard deviation: square root of variance
    std_dev = np.sqrt(variance)
    
    # Skewness: (third central moment) / std^3
    skewness = (moments_df["E[X^r]"].iloc[3] - 3 * moments_df["E[X^r]"].iloc[2] * mean + 2 * mean**3) / std_dev**3
    
    # Kurtosis: (fourth central moment) / std^4
    kurtosis = (moments_df["E[X^r]"].iloc[4] - 4 * moments_df["E[X^r]"].iloc[3] * mean + 6 * moments_df["E[X^r]"].iloc[2] * mean**2 - 3 * mean**4) / std_dev**4
    
    # Compile the statistics
    statistics = {
        "Mean": mean,
        "Variance": variance,
        "Standard Deviation": std_dev,
        "Skewness": skewness,
        "Kurtosis": kurtosis
    }
    
    return statistics

def moments_stat(mu1, sigma1, h, mu2,  sigma2, x0, xm, lambda1, lambda2):
    """
    Calculate moments and statistics for a mixture of log-normals.
    """
    # Validate input parameters
    if not (0 <= lambda1 <= 1 and 0 <= lambda2 <= 1):
        raise ValueError("lambda1 and lambda2 must be between 0 and 1.")
    if lambda1 + lambda2 != 1:
        raise ValueError("lambda1 + lambda2 must equal 1.")
    
    lambda_k = [lambda1, lambda2]
    mu_k = [mu1, mu2]
    sigma_k = [sigma1, sigma2]
    
    # Compute normalization constant
    Cnst = norm_C(x0, xm, mu1, sigma1, mu2, sigma2, h, lambda1=lambda1, lambda2=lambda2)
    logger.info(f"- Normalization constant C = {Cnst}")
    
    # Compute moments and statistics
    moments_df = compute_moments(lambda_k, mu_k, sigma_k, h, x0, xm, C=Cnst)
    logger.info(moments_df)
    
    stats = compute_statistics(lambda_k, mu_k, sigma_k, h, x0, xm, C=Cnst)
    logger.info(stats)
    
    # Get the first and second moments (mean and E[X^2])
    E_X = moments_df.loc["r=1", "E[X^r]"]  # E[X] is the first moment (r=1)
    E_X2 = moments_df.loc["r=2", "E[X^r]"]  # E[X^2] is the second moment (r=2)
    
    # Compute variance and standard deviation
    variance = E_X2 - E_X**2
    std_deviation = np.sqrt(variance)
    
    logger.info(f"- The standard deviation s = {std_deviation:.3f}")

