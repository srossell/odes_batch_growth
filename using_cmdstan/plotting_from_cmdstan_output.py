"""
Reading cmdstan outputs and plotting using arviz
"""
import os

import arviz as az
import matplotlib.pyplot as plt

idata = az.from_cmdstan(posterior='batch_growth_[0-9].csv')

# plot posterior
az.plot_posterior(idata, var_names=['y0'])
plt.show()


# Pair plots
az.plot_pair(idata, var_names=['y0', 'p'])
plt.show()


# Plotting traces
az.plot_trace(idata.posterior, var_names=['y0', 'p'])
plt.show()

