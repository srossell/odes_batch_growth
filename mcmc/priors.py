"""
Setting priors for MCMC parameter estimation of the parameters in the batch
growht model.

The maximum possible yield is if glucose was used entirely for biomass
synthesis. That is the maximum yield possible given element and charge
conservation.

The maximum rate of glucose consumption is calculated assuming a maximum yield,
and that the maximum growth rate cannot be higher than that of E. coli.
"""

import numpy as np
import scipy.stats as ss

########################################
# Biomass yield on sugar

Yxs_max = 0.139  # g_X/mmol_glc

Yxs_alpha = 2
Yxs_beta = 50
Yxs_x = np.linspace(0, 0.2, 100)
Yxs_y = ss.gamma.pdf(Yxs_x, a=Yxs_alpha, scale=1/Yxs_beta)

Yxs_stan = 'gamma({alpha}, {beta})'.format(alpha=Yxs_alpha, beta=Yxs_beta)

########################################
# growth rate. Less than E. coli (0.89 1/h)
mu_max = 0.89
mu_alpha = 2
mu_beta = 10
mu_x = np.linspace(0, 1, 100)
mu_y = ss.gamma.pdf(mu_x, a=mu_alpha, scale=1/mu_beta)

mu_stan = 'gamma({alpha}, {beta})'.format(alpha=mu_alpha, beta=mu_beta)

########################################
# Initial glucose. Known with a lot of certaintiy
glc0_mu = 100 # mmol/l
glc0_sd = 1
glc0_x = np.linspace(90, 110, 100)
glc0_y = ss.norm.pdf(glc0_x, glc0_mu, glc0_sd)

glc0_stan = 'normal({mu}, {sd})'.format(mu=glc0_mu, sd=glc0_sd)

########################################
# Initial biomass. We pitch 1 g DW, but are uncertain about the fraction of
# viable cells
x0_alpha = 2
x0_beta = 2
x0_x = np.linspace(0, 1, 100)
x0_y = ss.beta.pdf(x0_x, x0_alpha, x0_beta)

x0_stan = 'beta({alpha}, {beta})'.format(alpha=x0_alpha, beta=x0_beta)


########################################
# Stan model
stan_str = """
parameters {{
    real yxs;
    real mu;
    real glc0;
    real x0;
}}

model {{
    yxs ~ {yxs};
    mu ~ {mu};
    glc0 ~ {glc0};
    x0 ~ {x0};
}}

""".format(
        yxs=Yxs_stan,
        mu=mu_stan,
        glc0=glc0_stan,
        x0=x0_stan
        )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pystan

    # Compile stan model
    sm = pystan.StanModel(model_code=stan_str)
    # Sample posterior
    stan_fit = sm.sampling(iter=2000, chains=2, n_jobs=-1, warmup=1000)
    # extract samples
    fit_dict = stan_fit.extract()


    #######################################
    # PLOTS
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].hist(fit_dict['yxs'], density='kde', bins=50, alpha=0.5)
    ax[0, 0].plot(Yxs_x, Yxs_y)
    ax[0, 0].axvline(Yxs_max, color='C2', lw=2)
    ax[0, 0].set_xlabel('Yxs [g_X/mmol_glc]')

    ax[0, 1].hist(fit_dict['mu'], density='kde', bins=50, alpha=0.5)
    ax[0, 1].plot(mu_x, mu_y)
    ax[0, 1].axvline(mu_max, color='C2', lw=2)
    ax[0, 1].set_xlabel('$\mu$ [$h^{-1}$]')

    ax[1, 0].hist(fit_dict['glc0'], density='kde', bins=50, alpha=0.5)
    ax[1, 0].plot(glc0_x, glc0_y)
    ax[1, 0].set_xlabel('Initial glucose [mmol]')

    ax[1, 1].hist(fit_dict['x0'], density='kde', bins=50, alpha=0.5)
    ax[1, 1].plot(x0_x, x0_y)
    ax[1, 1].set_xlabel('Initial biomass [g]')
    fig.tight_layout()
    plt.show()

