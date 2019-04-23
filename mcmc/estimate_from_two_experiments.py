from scipy.integrate import solve_ivp
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan
import scipy.stats as ss

from mcmc.priors import Yxs_stan
from mcmc.priors import Yxs_x, Yxs_y
from mcmc.priors import glc0_stan
from mcmc.priors import glc0_x, glc0_y
from mcmc.priors import mu_stan
from mcmc.priors import mu_x, mu_y
from mcmc.priors import x0_stan
from mcmc.priors import x0_x, x0_y

from bcodes.ratevector import create_rate_vector
from bcodes.stoichiometrymatrix import build_stoichiometry_matrix
from model.batch_growth import batch_growth as m


###############################################################################
# INPUTS

t_odes = [0, 9]  # [t0 t_final]
t_sim = np.arange(1, 9.5, 0.5)  # Times to simulate after fitting
i_obs_1 = [8, 10, 12, 14]  # 5, 6, 7 and 8h indices for origi
i_obs_2 = [14, 16]  # 8 and 9h

inoc_factors = [1., 2.]  # factors mutliplying the initial pitch

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
    int<lower=1> K;
    int cum_sizes[K];
    real y[N, 2];
    real t0;
    real t_obs[N];
    real inoc_factors[2];
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
}}

model {{
    real y_hat[N, 2];
    real y00[2];
    int pos;
    pos = 1;
    sigma ~ normal(0, 1);
    p[1] ~ {mu};
    p[2] ~ {yxs};
    y0[1] ~ {glc0};
    y0[2] ~ {x0};
    y00[1] = y0[1];
    for (k in 1:K) {{
        y00[2] = y0[2] * inoc_factors[k];
        y_hat[pos: cum_sizes[k]] = integrate_ode_rk45(myodes, y00, t0, t_obs[pos: cum_sizes[k]], p, x_r, x_i);
        pos = pos + cum_sizes[k];
        }}
    for (t in 1:cum_sizes[1]) {{
        y[t, 1] ~ normal(y_hat[t, 1], sigma);
        y[t, 2] ~ normal(y_hat[t, 2], sigma);
        }}
    // Only dry weight was measured in the second experiment
    for (t in (cum_sizes[1] + 1):N) {{
        y[t, 2] ~ normal(y_hat[t, 2], sigma);
        }}
}}
""".format(
        yxs=Yxs_stan,
        mu=mu_stan,
        glc0=glc0_stan,
        x0=x0_stan
        )

###############################################################################
# FUNCTIONS

def odes(t, y):
    return np.dot(S, v(y))


def select_and_add_noise(i_obs, sol):
    # Collecting observations and adding noise to them
    obs = sol.y.T[i_obs]
    t_obs = t_sim[i_obs]

    dw_err = ss.norm.rvs(loc=0, scale=0.5, size=len(t_obs), random_state=42)
    glc_err = ss.norm.rvs(loc=0, scale=2, size=len(t_obs), random_state=42)

    obs[:, 0] += glc_err
    obs[:, 1] += dw_err
    return t_obs, obs


def hpd_shade(ax, interval, samples):
    """
    Caculate high prob density intervals using arviz and fill in between the
    interval.

    ACCEPTS
    ax [matplotlib axes]
    interval [float] fraction of the hpd localization
    samples [2d array] posterior samples. Columns are time points, rows samples
    """
    my_hpd = az.hpd(samples, credible_interval=interval)
    ax.fill_between(datadict['t_sim'], my_hpd[:, 0], my_hpd[:, 1], alpha=0.2,
            color='C1')


def hpd_shades(ax, interval_list, samples):
    for interval in interval_list:
        hpd_shade(ax, interval, samples)


def get_kde(data, x):
    kernel = ss.gaussian_kde(data)
    kernel_values = kernel(x)
    return kernel_values / kernel_values.max()


###############################################################################
# STATEMENTS

# Run simulations
S = build_stoichiometry_matrix(m.id_sp, m.id_rs, m.mass_balances)

v = create_rate_vector(m.id_sp, m.id_rs, m.rates, m.params)

sol_1 = solve_ivp(odes, t_odes, m.init, method='LSODA', vectorized=True,
        t_eval=t_sim)

sol_2 = solve_ivp(odes, t_odes,
        [
            m.init[0],
            m.init[1] * inoc_factors[1]
            ],
        method='LSODA', vectorized=True,
        t_eval=t_sim)


t_obs_1, obs_1 = select_and_add_noise(i_obs_1, sol_1)
t_obs_2, obs_2 = select_and_add_noise(i_obs_2, sol_2)

obs = np.concatenate((obs_1, obs_2))
t_obs = np.concatenate((t_obs_1, t_obs_2))
cum_sizes = np.array([len(t_obs_1), len(t_obs_2)]).cumsum()

# Dictionary to pass to the stan model
datadict = {}
datadict['N'] = obs.shape[0]  # number of samples
datadict['y'] = obs
datadict['t0'] = 0
datadict['cum_sizes'] = cum_sizes
datadict['K'] = len(cum_sizes)
datadict['t_obs'] = t_obs
datadict['inoc_factors'] = inoc_factors

# compile StanModel
sm = pystan.StanModel(model_code=stan_str)

# run mcmc
fit = sm.sampling(
                    data=datadict,
                    iter=7000,
                    chains=4,
                    n_jobs=4,
                    warmup=2000,
                    algorithm='NUTS',
                    seed = 42
                    )

df_summary =  pd.DataFrame(
    summary['summary'],
    columns=summary['summary_colnames'],
    index=summary['summary_rownames']
)

fit_dict= fit.extract()
fit_df = fit.to_dataframe()

params_post = fit_df[['y0[1]', 'y0[2]','p[1]', 'p[2]']]
params_post.columns = ['glc_0', 'dw_0', 'mu_max', 'Yxs']


###############################################################################
# OUTPUTS

print(df_summary.head(6))

# Pair plots
fig, ax = plt.subplots()
pd.plotting.scatter_matrix(params_post.iloc[::10], diagonal='kde', ax=ax)
fig.tight_layout()
plt.show()


# Plot traces and density
nrows = len(var_dict)
ncols = 3

fig = plt.figure(figsize=(10, 10))
ax = np.zeros((nrows, ncols)).astype(object)
for i, key in enumerate(var_dict):
    ax[i, 0] = plt.subplot2grid((nrows, ncols), (i, 0))
    az.plot_posterior(fit_df[key].values, ax=ax[i, 0], textsize=12,
            point_estimate='mean')
    ax[i, 0].set_title('')
    ax[i, 0].set_xlabel(var_dict[key])

    ax[i, 1] = plt.subplot2grid((nrows, ncols), (i, 1), colspan=2)
    fit_df.groupby('chain').plot(x='draw', y=key, ax=ax[i, 1], alpha=0.3, lw=1,
        legend=False)
    ax[i, 1].set_xlabel('')
    ax[i, 1].set_ylabel(var_dict[key])
fig.tight_layout()
plt.show()


# Scatter plot showing correlations
fig, ax = plt.subplots()
sc = ax.scatter(x=fit_df['p[1]'], y=fit_df['y0[2]'], c=fit_df['p[2]'],
        alpha=0.2, s=3)
ax.set_xlabel('$\mu^{max}$ [$h^{-1}$]')
ax.set_ylabel('initial biomass [g]')
cbar = plt.colorbar(sc)
plt.show()


# Comparing prior and posterior probability distributions
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
ax[0, 0].plot(Yxs_x, Yxs_y / Yxs_y.max())
ax[0, 0].plot(Yxs_x, get_kde(params_post['Yxs'], Yxs_x))
ax[0, 0].set_xlabel('Yxs')
ax[0, 1].plot(mu_x, mu_y / mu_y.max(), label='prior')
ax[0, 1].plot(mu_x, get_kde(params_post['mu_max'], mu_x), label='posterior')
ax[0, 1].set_xlabel('$\mu^{max}$')
ax[0, 1].legend()
ax[1, 0].plot(glc0_x, glc0_y / glc0_y.max())
ax[1, 0].plot(glc0_x, get_kde(params_post['glc_0'], glc0_x))
ax[1, 0].set_xlabel('initial glcuose')
ax[1, 1].plot(x0_x, x0_y / x0_y.max())
ax[1, 1].plot(x0_x, get_kde(params_post['dw_0'], x0_x))
ax[1, 1].set_xlabel('initial biomass')
fig.tight_layout()
plt.show()

