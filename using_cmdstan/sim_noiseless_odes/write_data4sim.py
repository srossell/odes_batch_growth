"""
Write data dict as R object to be used as input for cmdstan
"""
import os

import pystan

from model.simulate_batch_growth import t_sim
from model.batch_growth import batch_growth as m

###############################################################################
# INPUTS

script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')
f_stan_input_data = os.path.join(
        output_files_path, 'batch_growth_sim_input_data.r'
        )


###############################################################################
# STATEMENTS

# Dictionary to pass to the stan model
datadict = {}
datadict['t0'] = 0
datadict['t_sim'] = t_sim
datadict['T'] = len(t_sim)
datadict['y0'] = m.init
datadict['p'] = []

pystan.misc.stan_rdump(datadict, f_stan_input_data)

