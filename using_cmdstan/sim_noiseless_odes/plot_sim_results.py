import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from model.simulate_batch_growth import t_sim


script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')
sim_output_file = os.path.join(
        output_files_path, 'batch_growth_sim_output.r'
        )

idata = az.from_cmdstan(sim_output_file)

post = idata.posterior

# Changing coordinates to represent the simulation time
post = post.assign_coords(y_hat_dim_0=t_sim)

# Given variable names in the coordinates
post = post.assign_coords(y0_dim_0=['glc', 'dw'])

fig, ax = plt.subplots(ncols=2)
ax[0].plot(t_sim, post.isel(chain=0, draw=0, y_hat_dim_1=0).y_hat, marker='o')
ax[1].plot(t_sim, post.isel(chain=0, draw=0, y_hat_dim_1=1).y_hat, marker='o')
plt.show()


