# Aerosol-PSD-Modeling

Aerosol PSD statistical modeling: time-variant, time-invariant, and change-point detection


* `convex_log.py`  Performs parameter estimation by minimizing the sum of squared errors (SSE) to fit a convex combination of two lognormal probability density functions. Returns the optimal values of \$\mu\_1\$, \$\sigma\_1\$, \$\mu\_2\$, \$\sigma\_2\$, and \$h\$ for a given time slice or dataset.

* `gradients_fx.py` Computes the diameter gradient \$\partial f/\partial d\$ of the fitted probability density function. This function helps assess how particle number concentration changes with respect to particle size, revealing structural behavior in PSD.

* `mmts_stat.py` Calculates the zeroth to fourth-order statistical moments of the fitted PDF: total mass (area under the curve), mean diameter, variance, skewness, and kurtosis. These are essential for characterizing the distributional properties of aerosol particles.


