"""
Write a stan model into a file
"""

import os

from bcodes.utils.stanodes import create_stan_odes_str
from mcmc.priors import Yxs_stan
from mcmc.priors import Yxs_x, Yxs_y
from mcmc.priors import glc0_stan
from mcmc.priors import glc0_x, glc0_y
from mcmc.priors import mu_stan
from mcmc.priors import mu_x, mu_y
from mcmc.priors import x0_stan
from mcmc.priors import x0_x, x0_y
from model.batch_growth import batch_growth as m


###############################################################################
# INPUTS

script_path = os.path.split(__file__)[0]
output_files_path = os.path.join(script_path, 'output_files')
f_stan_model_str = os.path.join(output_files_path, 'batch_growth.stan')

params2fit = ['mu_max', 'Yxs']

odes_str, trans_dict, rates = create_stan_odes_str(
        m.id_sp, m.id_rs, m.rates, m.mass_balances, m.params, params2fit)



stan_str = """
functions {{
    {odes_str}
}}

data {{
    int<lower=1> N;
    int<lower=1> T;
    real y[N, 2];
    real t0;
    real t_obs[N];
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
    real y_hat[N, 2];
    y_hat = integrate_ode_bdf(
        odes, y0, t0, t_obs, p, x_r, x_i, 1e-12, 1e-12, 1e8
        );
}}

model {{
    sigma ~ normal(0, 1);
    p[1] ~ {mu};
    p[2] ~ {yxs};
    y0[1] ~ {glc0};
    y0[2] ~ {x0};
    for (t in 1:N)
        y[t] ~ normal(y_hat[t], sigma);
}}

generated quantities {{
    real y_hat_n[T, 2];
    real y_hat_sigma[T, 2];

    y_hat_n = integrate_ode_rk45(
        odes, y0, t0, t_sim, p, x_r, x_i, 1e-12, 1e-12, 1e8
        );
    // Add error with estimated sigma
    for (i in 1:T) {{
        y_hat_sigma[i, 1] = y_hat_n[i, 1] + normal_rng(0, sigma[1]);
        y_hat_sigma[i, 2] = y_hat_n[i, 2] + normal_rng(0, sigma[2]);
        }}
}}
""".format(
        odes_str=odes_str,
        yxs=Yxs_stan,
        mu=mu_stan,
        glc0=glc0_stan,
        x0=x0_stan
        )



###############################################################################
# OUTPUTS


if __name__ == '__main__':
    with open(f_stan_model_str, 'w') as f:
        f.write(stan_str)

