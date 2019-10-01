"""
Write a stan model into a file
"""

import os

from mcmc.priors import Yxs_stan
from mcmc.priors import Yxs_x, Yxs_y
from mcmc.priors import glc0_stan
from mcmc.priors import glc0_x, glc0_y
from mcmc.priors import mu_stan
from mcmc.priors import mu_x, mu_y
from mcmc.priors import x0_stan
from mcmc.priors import x0_x, x0_y


###############################################################################
# INPUTS

script_path = os.path.split(__file__)[0]
f_stan_model_str = os.path.join(script_path, 'batch_growth.stan')


var_dict = {
        'p[1]': 'mu_max',
        'p[2]': 'Yxs',
        'y0[1]':'glc_0',
        'y0[2]': 'X_0',
        }

stan_str = """
functions {{
    real[] myodes(
        real t,
        real[] y,
        real[] p,
        real[] x_r,
        int[] x_i
        )
        {{
            real dydt[2];
            dydt[1] = -y[2] * (p[1] / p[2]) * (y[1]/1) / (1 + (y[1]/1));
            dydt[2] = (y[2] * p[1] * (y[1]/1) / (1 + (y[1]/1)));
            return dydt;
        }}
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
    y_hat = integrate_ode_rk45(myodes, y0, t0, t_obs, p, x_r, x_i);
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

    y_hat_n = integrate_ode_rk45(myodes, y0, t0, t_sim, p, x_r, x_i);
    // Add error with estimated sigma
    for (i in 1:T) {{
        y_hat_sigma[i, 1] = y_hat_n[i, 1] + normal_rng(0, sigma[1]);
        y_hat_sigma[i, 2] = y_hat_n[i, 2] + normal_rng(0, sigma[2]);
        }}
}}
""".format(
        yxs=Yxs_stan,
        mu=mu_stan,
        glc0=glc0_stan,
        x0=x0_stan
        )



###############################################################################
# OUTPUTS

with open(f_stan_model_str, 'w') as f:
    f.write(stan_str)
