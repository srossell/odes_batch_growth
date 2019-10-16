"""
Write data dict as R object to be used as input for cmdstan
"""
import os

import numpy as np

import pystan

from affinity_titration.sim_data import obs
from affinity_titration.sim_data import t_obs
from affinity_titration.sim_data import Kms

###############################################################################
# INPUTS

script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')
f_stan_input_data = os.path.join(
        output_files_path, 'affinity_titration_input_data.r'
        )
# long, and there are len(Kms) of them
cum_sizes = np.repeat(int(len(t_obs)/len(Kms)), len(Kms)).cumsum()

t_sim = np.arange(1, 8.5, 0.5)

###############################################################################
# STATEMENTS

# Dictionary to pass to the stan model
datadict = {}
datadict['N'] = obs.shape[0]  # number of samples
datadict['y'] = obs
datadict['t0'] = 0
datadict['cum_sizes'] = cum_sizes
datadict['K'] = len(cum_sizes)
datadict['t_sim'] = t_sim
datadict['T'] = len(t_sim)
datadict['t_obs'] = t_obs
datadict['kms']= np.array(Kms)[:, np.newaxis]

if __name__ == '__main__':
    pystan.misc.stan_rdump(datadict, f_stan_input_data)

