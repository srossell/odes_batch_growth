"""
Toy model of a microbial culture growing in a batch fermentation.
"""

class BatchGrowth(object):
    def __init__(self, id_sp, id_rs, rates, mass_balances, params, init):
        self.id_sp = id_sp
        self.id_rs = id_rs
        self.rates = rates
        self.mass_balances = mass_balances
        self.params = params
        self.init = init

id_sp = [
        'glc',  # mmol
        'dw'  # g DW
        ]

id_rs = [
        'v_glc',  # mmol/h  # glucose consumption rate (q_glc * dw)
        'v_bio',  # gDW / h  # growth rate
        ]

rates = {
        'v_glc': 'dw * mu_max / Yxs * (glc/bw) / (Km_q_glc + (glc/bw))',
        'v_bio': 'dw * mu_max * (glc/bw) / (Km_q_glc + (glc/bw))'
        }

params = {
        'Yxs': 0.09,  # mmol_glc/gDW  # calculated from 0.5 g/g
        'Km_q_glc': 1,  # mmol/kg_bw
        'mu_max': 0.35,  # 1/h
        'bw': 1,  # kg  # broth weight
        }

mass_balances = {
        'glc': {'v_glc': -1},
        'dw': {'v_glc': params['Yxs']}
        }

init = [100, 0.2]

batch_growth = BatchGrowth(id_sp, id_rs, rates, mass_balances, params, init)

