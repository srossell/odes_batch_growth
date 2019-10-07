"""
Reading cmdstan outputs and plotting using arviz
"""
import os

import arviz as az
import matplotlib.pyplot as plt

idata = az.from_cmdstan(posterior='batch_growth_[0-9].csv')
post = idata.posterior

# Changing the coordinates in th xarray
post = post.assign_coords(y0_dim_0=['glc0', 'dw0'])
post = post.assign_coords(p_dim_0=['mu_max', 'Yxs'])

# plot posterior
az.plot_posterior(post, var_names=['y0'])
plt.show()


# Pair plots
az.plot_pair(post, var_names=['y0', 'p'])
plt.show()


# Plotting traces
az.plot_trace(post, var_names=['y0', 'p'])
plt.show()

