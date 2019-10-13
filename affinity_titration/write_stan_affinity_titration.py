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
f_stan_model_str = os.path.join(output_files_path, 'affinity_titration.stan')

params2fit = ['mu_max', 'Yxs']
params2tune = ['Km_q_glc']

odes_str, trans_dict, rates = create_stan_odes_str(
        m.id_sp, m.id_rs, m.rates, m.mass_balances, m.params, params2fit,
        params2tune)

stan_str = """
functions {{
    {odes_str}
}}

data {{
    int<lower=1> N;
    int<lower=1> T;
    real y[N, {len_idsp}];
    real t0;
    real t_obs[N];
    real t_sim[T];
    real kms[{len_p2tune}];
    }}

transformed data {{
    real x_r[{len_p2tune}] = kms;
    int x_i[0];
}}

parameters {{
    real<lower=0> y0[{len_idsp}]; // init
    vector<lower=0>[{len_idsp}] sigma;
    real<lower=0> p[{len_p2fit}];
}}

transformed parameters {{
    real y_hat[N, {len_idsp}];
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
    real y_hat_n[T, {len_idsp}];
    real y_hat_sigma[T, {len_idsp}];

    y_hat_n = integrate_ode_rk45(
        odes, y0, t0, t_sim, p, x_r, x_i, 1e-12, 1e-12, 1e8
        );
    // Add error with estimated sigma  // NOTE not general
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
        x0=x0_stan,
        len_idsp=len(m.id_sp),
        len_p2fit=len(params2fit),
        len_p2tune=len(params2tune)
        )



###############################################################################
# OUTPUTS


if __name__ == '__main__':
    with open(f_stan_model_str, 'w') as f:
        f.write(stan_str)

