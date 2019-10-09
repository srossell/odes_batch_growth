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

################################################################################
################################################################################
# TODO work this into a function, and clean up.
# TODO plot using t_sim

# Extracting one variable time course with all its 5000 samples

# Get the data array
y_hat_n = idata.posterior.y_hat_n

# (optional) select the first 100 draws (for playing around
# y = y_hat_n.isel(draw=slice(0, 100))
y = y_hat_n

# Select the first variable
y1 = y.sel(y_hat_n_dim_1=0)

# use pandas to shuffle the data array
y1df = y1.to_dataframe()
y1df = y1df.drop('y_hat_n_dim_1', axis=1)
y1df = y1df.reset_index(['chain', 'draw'])
y1df.loc[:, ['chain', 'draw']] =y1df.loc[:, ['chain', 'draw']].astype(str)
y1df['cd'] = y1df['chain'].str.cat(y1df['draw'])

# Calclate the high probability density intervals
my_hpd = az.hpd(y1df.pivot(columns='cd', values='y_hat_n').values.T)
################################################################################
################################################################################


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

