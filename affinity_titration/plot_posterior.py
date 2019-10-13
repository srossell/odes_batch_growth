import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.simulate_batch_growth import t_obs, t_sim


script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')


idata = az.from_cmdstan(
        posterior=os.path.join(
            output_files_path,
            'affinity_titration_chain_[0-9].csv'
            )
        )

post = idata.posterior


# Changing coordinates
post = post.rename({
    'y0_dim_0': 'initial',
    'y_hat_dim_1': 'var_obs',
    'y_hat_dim_0': 'time_obs',
    'y_hat_n_dim_1': 'var_sim',
    'y_hat_n_dim_0': 'time_sim',
    })

post = post.assign_coords(initial=['glc0', 'dw0'])
post = post.assign_coords(p_dim_0=['mu_max', 'Yxs'])
post = post.assign_coords(var_sim=['glc', 'dw'])
post = post.assign_coords(var_obs=['glc', 'dw'])
post = post.assign_coords(time_obs=t_obs)
post = post.assign_coords(sime_sim=t_sim)

# Stacking chain and draw
post_stack = post.stack({'cd': ['chain', 'draw']})

# high probability density intervals
hpd_glc = az.hpd(post_stack.y_hat_n.sel(var_sim='glc').T)
hpd_dw = az.hpd(post_stack.y_hat_n.sel(var_sim='dw').T)

# Pair plots
az.plot_pair(post, var_names=['y0', 'p'])
plt.show()

# Traces
az.plot_trace(post, var_names=['y0', 'p'])
plt.show()

# Posterior
az.plot_posterior(post, var_names=['y0', 'p'])
plt.show()

# Scatter
plt.scatter(
        x=post_stack.y0.loc['dw0'],
        y=post_stack.p.loc['mu_max'],
        c=post_stack.p.loc['Yxs'])
plt.show()

# hpd
fig, ax = plt.subplots(ncols=2)
ax[0].fill_between(t_sim, hpd_glc[:, 0], hpd_glc[:, 1], alpha=0.2)
ax[1].fill_between(t_sim, hpd_dw[:, 0], hpd_dw[:, 1], alpha=0.2)
plt.show()

