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
from affinity_titration.sim_data import Kms

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
    int<lower=1> K;
    int cum_sizes[K];
    real y[N, {len_idsp}];
    real t0;
    real t_obs[N];
    real t_sim[T];
    real kms[{len_p2tune_vals}, {len_p2tune}];
    }}

transformed data {{
    real x_r[{len_p2tune_vals}, {len_p2tune}];
    int x_i[0];
}}

parameters {{
    real<lower=0> y0[{len_idsp}]; // init
    vector<lower=0>[{len_idsp}] sigma;
    real<lower=0> p[{len_p2fit}];
}}

transformed parameters {{
}}

model {{
    real y_hat[N, {len_idsp}];
    int pos;
    pos = 1;
    sigma ~ normal(0, 1);
    p[1] ~ {mu};
    p[2] ~ {yxs};
    y0[1] ~ {glc0};
    y0[2] ~ {x0};
    for (k in 1:K) {{
        y_hat[pos: cum_sizes[k]] = integrate_ode_bdf(odes, y0, t0, t_obs[pos: cum_sizes[k]], p, kms[k], x_i, 1e-12, 1e-12, 1e8);
        for (t in pos: cum_sizes[k])
            y[t] ~ normal(y_hat[t], sigma);
        pos = pos + cum_sizes[1];
    }}
}}

generated quantities {{
    real y_sim[T * {len_p2tune_vals}, {len_idsp}];
    int pos_0; // begin
    int pos_1; // end
    pos_0 = 1;
    pos_1 = T;
    for (k in 1:K) {{
        y_sim[pos_0: pos_1] = integrate_ode_bdf(odes, y0, t0, t_sim, p, kms[k], x_i, 1e-12, 1e-12, 1e8);
        pos_0 = pos_0 + T;
        pos_1 = pos_1 + T;
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
        len_p2tune=len(params2tune),
        len_p2tune_vals=len(Kms)
        )



###############################################################################
# OUTPUTS


if __name__ == '__main__':
    with open(f_stan_model_str, 'w') as f:
        f.write(stan_str)

