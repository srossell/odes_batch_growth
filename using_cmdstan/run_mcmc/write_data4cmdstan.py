"""
Write data dict as R object to be used as input for cmdstan
"""
import os

import pystan

from model.simulate_batch_growth import obs
from model.simulate_batch_growth import t_obs
from model.simulate_batch_growth import t_sim

###############################################################################
# INPUTS

script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')
f_stan_input_data = os.path.join(
        output_files_path, 'batch_growth_input_data.r'
        )


###############################################################################
# STATEMENTS

# Dictionary to pass to the stan model
datadict = {}
datadict['N'] = obs.shape[0]  # number of samples
datadict['y'] = obs
datadict['t0'] = 0
datadict['t_sim'] = t_sim
datadict['T'] = len(t_sim)
datadict['t_obs'] = t_obs

if __name__ == '__main__':
    pystan.misc.stan_rdump(datadict, f_stan_input_data)

