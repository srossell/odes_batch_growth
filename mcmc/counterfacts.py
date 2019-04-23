"""
Counterfactual simulations. Having made a first experiment, we updated our
beliefs on the distribution of parameter values. Now we ask ourselves which
experiment should we perform next. If we can only measure two points in time
and one variable, what would we choose.

In our yeast cake example, we imagine we are deciding between halving or
doubling the size of the inoculum (the initial amount of biomass). Then we
assume we can only take two samples from which we can measure EITHER the
glucose content or the dry biomass.

From the simulations in this script, I would conclude that it would be best to
measure later time points and that it would be more informative to measure dry
weight than glucose. As examples of less informative experiments we would have:
    i) measure glucose early (worst experiment)
    ii) measure dw early
    iii) measure glucose late.
    iv) measure dw late with lower inoculum

Now, how should we measure performance. We  will still assume that we cannot
know the parameters. One way that occurs to me is extent to which the
uncertainties are reduced after the new observations are made. I still need to
think about how to quantify that, maybe the volume of the normalized
parameters.
"""

from scipy.integrate import solve_ivp
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcodes.ratevector import create_rate_vector
from bcodes.stoichiometrymatrix import build_stoichiometry_matrix
from bcodes.jacobian import build_jacobian
from model.batch_growth import batch_growth as m

###############################################################################
# INPUTS

params_post = pd.read_pickle('params_post.pkl')

t_odes = [0, 12]  # [t0 t_final]
t_sim = np.arange(1, 9.1, 0.1)

S = build_stoichiometry_matrix(m.id_sp, m.id_rs, m.mass_balances)

params = m.params.copy()
params['mu_max'] = 'p[0]'
params['Yxs'] = 'p[1]'


v = create_rate_vector(m.id_sp, m.id_rs, m.rates, params, p_is_arg=True)

def odes(t, y, p):
    return np.dot(S, v(y, p))


trans_dict = dict(
        **params,
        **dict(zip(
            m.id_sp,
            ['y[{}]'.format(i) for i, sp in enumerate(m.id_sp)]
            ))
        )

jac = build_jacobian(m.id_sp, m.id_rs, m.rates, m.mass_balances,
        list(params.keys()), trans_dict)


n_samples = 1000
params_sample = params_post.sample(n_samples)

down_list = []
up_list = []
for i in params_sample.index:
    sol_down = solve_ivp(
            lambda t, y: odes(
                t, y,
                [
                    params_post.loc[i, 'mu_max'],
                    params_post.loc[i, 'Yxs']
                ]
                ),
            t_odes,
            [
                params_post.loc[i, 'glc_0'],
                params_post.loc[i, 'dw_0'] * 0.5
            ],
            method='LSODA', vectorized=True,# t_eval=t_sim,
            jac=lambda t, y: jac(
                t, y,
                p = [
                    params_post.loc[i, 'mu_max'],
                    params_post.loc[i, 'Yxs']
                ]
                ),
            t_eval=t_sim
            )
    down_list.append(pd.DataFrame(sol_down.y.T, columns=['glc', 'dw'],
        index=t_sim))

    sol_up = solve_ivp(
            lambda t, y: odes(
                t, y,
                [
                    params_post.loc[i, 'mu_max'],
                    params_post.loc[i, 'Yxs']
                ]
                ),
            t_odes,
            [
                params_post.loc[i, 'glc_0'],
                params_post.loc[i, 'dw_0'] * 2.0
            ],
            method='LSODA', vectorized=True,# t_eval=t_sim,
            jac=lambda t, y: jac(
                t, y,
                p = [
                    params_post.loc[i, 'mu_max'],
                    params_post.loc[i, 'Yxs']
                ]
                ),
            t_eval=t_sim
           )
    up_list.append(pd.DataFrame(sol_up.y.T, columns=['glc', 'dw'],
        index=t_sim))


df_down = pd.concat(down_list, keys=params_sample.index)
df_down = df_down.reset_index(0).pivot(columns='level_0')

df_up = pd.concat(up_list, keys=params_sample.index)
df_up = df_up.reset_index(0).pivot(columns='level_0')

intervals = [0.50, 0.75, 0.95]
def hpd_shade(samples, interval, ax):
    my_hpd = az.hpd(samples, credible_interval=interval)
    ax.fill_between(t_sim, my_hpd[:, 0], my_hpd[:, 1], alpha=0.3,
            color='C1')

def hpd_shades(samples, interval_list, ax):
    for interval in interval_list:
        hpd_shade(samples, interval, ax)

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row',
        figsize=(10, 6))
ax[0, 0].set_title('Higher inoculum')
hpd_shades(df_up['glc'].values.T, intervals, ax=ax[0, 0])
hpd_shades(df_up['dw'].values.T, intervals, ax=ax[1, 0])
ax[0, 0].set_ylabel('Glucose (mmol)')
ax[1, 0].set_ylabel('Dry weight (g)')

ax[0, 1].set_title('Lower inoculum')
hpd_shades(df_down['glc'].values.T, intervals, ax=ax[0, 1])
hpd_shades(df_down['dw'].values.T, intervals, ax=ax[1, 1])
ax[1, 0].set_xlabel('time (h)')
ax[1, 1].set_xlabel('time (h)')
fig.tight_layout()
plt.show()

