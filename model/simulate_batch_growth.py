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

t_odes = [0, 9]  # [t0 t_final]
t_sim = np.arange(1, 9.5, 0.5)  # Times to simulate after fitting
i_obs = [8, 10, 12, 14]  # 5, 6, 7 and 8h indices for the observed measurements


###############################################################################
# STATEMENTS

S = build_stoichiometry_matrix(m.id_sp, m.id_rs, m.mass_balances)

v = create_rate_vector(m.id_sp, m.id_rs, m.rates, m.params)

def odes(t, y):
    return np.dot(S, v(y))

sol = solve_ivp(odes, t_odes, m.init, method='LSODA', vectorized=True,
        t_eval=t_sim)

# Collecting observations and adding noise to them
obs = sol.y.T[i_obs]
t_obs = t_sim[i_obs]

dw_err = ss.norm.rvs(loc=0, scale=0.5, size=len(t_obs), random_state=42)
glc_err = ss.norm.rvs(loc=0, scale=2, size=len(t_obs), random_state=42)

obs[:, 0] += glc_err
obs[:, 1] += dw_err

if __name__ == '__main__':
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
    ax[0].plot(sol.t, sol.y.T[:, 0])
    ax[0].scatter(t_obs, obs[:, 0], color='k')
    ax[0].set_ylabel('Glucose (mmol)')
    ax[1].plot(sol.t, sol.y.T[:, 1])
    ax[1].scatter(t_obs, obs[:, 1], color='k')
    ax[1].set_ylabel('Dry weight (g)')
    ax[0].set_xlabel('Time (h)')
    ax[1].set_xlabel('Time (h)')
    fig.tight_layout()
    plt.show()

