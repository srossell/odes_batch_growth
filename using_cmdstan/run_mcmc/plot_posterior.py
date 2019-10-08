import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from model.simulate_batch_growth import t_sim


script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')


idata = az.from_cmdstan(
        posterior=os.path.join(output_files_path,'batch_growth_[0-9].csv')
        )

post = idata.posterior

# Changing the coordinates in th xarray
post = post.assign_coords(y0_dim_0=['glc0', 'dw0'])
post = post.assign_coords(p_dim_0=['mu_max', 'Yxs'])


# Pair plots
az.plot_pair(post, var_names=['y0', 'p'])
plt.show()

# Traces
az.plot_trace(post, var_names=['y0', 'p'])
plt.show()

# Posterior
az.plot_posterior(post, var_names=['y0', 'p'])
plt.show()

