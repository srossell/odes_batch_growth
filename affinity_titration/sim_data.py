"""
Simulate a titration of Km values (e.g. by adding a competitive inhibitor),
adding Gaussian noise to the observations, and then adding a few outliers.
"""
from itertools import count, islice

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

from bcodes.ratevector import create_rate_vector
from bcodes.stoichiometrymatrix import build_stoichiometry_matrix
from model.batch_growth import batch_growth as m

###############################################################################
# INPUTS
np.random.seed(seed=2357)

t_odes = [0, 9]  # [t0 t_final]
t_sim = [1, 1.5, 2, 2.5, 3, 5, 8]  # Times to simulate after fitting
i_obs = [8, 10, 12, 14]  # 5, 6, 7 and 8h indices for the observed measurements

params = m.params.copy()
params['Km_q_glc'] = 'p[0]'
id_sp = m.id_sp.copy()

Kms = [1, 5, 10, 25, 50, 100]

################################################################################
# STATEMENTS

S = build_stoichiometry_matrix(m.id_sp, m.id_rs, m.mass_balances)
v = create_rate_vector(m.id_sp, m.id_rs, m.rates, params, p_is_arg=True)

def odes(t, y, p):
    return np.dot(S, v(y, p))

# Run the integration of the differential equations with different affinities
sol_dict = {}
for p in Kms:
    sol_dict[p] = solve_ivp(
            lambda t, y: odes(t, y, [p]),
            [0, 9],
            m.init,
            method='LSODA',
            t_eval=t_sim)

# Concatenating simulations into an observations array
for k in Kms:
    try:
        sims = np.concatenate((sims, sol_dict[k].y.T))
    except NameError:
        sims = sol_dict[k].y.T

# Adding normal noise
obs = sims.copy()
glc_err = ss.norm.rvs(loc=0, scale=1, size=len(obs))
dw_err = ss.norm.rvs(loc=0, scale=0.05, size=len(obs))

obs[:, 0] += glc_err
obs[:, 1] += dw_err

# Simulating 5 outliers for each variable
outliers_vec = []
while len(outliers_vec) <= 10:
    draw = ss.uniform.rvs(loc=0, scale=1) * 2 - 1
    if (abs(draw) > 0.2) and (abs(draw) < 0.3):
        outliers_vec.append(draw)

outliers_array = np.zeros(len(t_sim)*len(Kms)*len(id_sp))
outliers_array.put(
        np.random.choice(
            np.arange(len(t_sim)*len(Kms)*len(m.id_sp)),
            len(outliers_vec)),
        outliers_vec
        )
outliers_array = outliers_array.reshape(-1, len(id_sp))
outliers_array = np.dot(outliers_array, np.diag([0.2, 1])) + 1

obs = obs * outliers_array
obs[obs < 0] = 0  # making sure no negative amounts are among the obs

t_obs = np.repeat(t_sim, len(Kms))


if __name__ == '__main__':
    # Creating indices to slice obs by experiment (Km value)
    low_ind = np.array(list(islice(count(0, len(t_sim)), len(Kms))))
    high_ind = low_ind + len(t_sim)

    # color list
    c = ['C'+ str(i) for i in range(10)]
    # Ploting observations
    fig, ax = plt.subplots(ncols=2)
    for i, l, h in zip(range(len(low_ind)), low_ind, high_ind):
        ax[0].plot(t_sim, obs[l:h, 0], 'o', c=c[i])
        ax[0].plot(t_sim, sims[l:h, 0], c=c[i])
        ax[1].plot(t_sim, obs[l:h, 1], 'o', c=c[i])
        ax[1].plot(t_sim, sims[l:h, 1], c=c[i])
        ax[0].set_ylim(0, 120)
        ax[1].set_ylim(0, 3.5)
    plt.show()


