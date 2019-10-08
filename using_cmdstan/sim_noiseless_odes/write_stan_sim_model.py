"""
Write a stan model into a file
"""

import os

from model.batch_growth import batch_growth as m
from bcodes.utils.stanodes import create_stan_odes_str

###############################################################################
# INPUTS

script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')
f_stan_model_str = os.path.join(output_files_path, 'batch_growth_sim.stan')

# NOTE here we are not going to estimate parameters
odes_str, trans_dict, rates = create_stan_odes_str(
        m.id_sp, m.id_rs, m.rates, m.mass_balances, m.params, [])


stan_str = """
functions {{
    {odes_str}
}}

data {{
    int<lower=1> T;
    real t0;
    real t_sim[T];
    }}

transformed data {{
    real x_r[0];
    int x_i[0];
}}

parameters {{
    real<lower=0> y0[2]; // init
    vector<lower=0>[2] sigma;
    real<lower=0> p[2];
}}

transformed parameters {{
    real y_hat[T, 2];
    y_hat = integrate_ode_bdf(odes, y0, t0, t_sim, p, x_r, x_i, 1e-12, 1e-12, 1e8);
}}

model {{
}}

generated quantities {{
        }}
""".format(
        odes_str=odes_str,
        )



###############################################################################
# OUTPUTS

with open(f_stan_model_str, 'w') as f:
    f.write(stan_str)

