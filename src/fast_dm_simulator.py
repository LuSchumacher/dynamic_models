import numpy as np
from numba import njit
from generative_models import diffusion_trial

def fast_dm_simulate(params, context):
    n_obs = context.shape[0]
    pred_rt = np.empty(n_obs)
    for t in range(n_obs):
        drift = np.random.normal(params[context[t]], params[6])
        ndt = np.random.uniform(params[5] - params[7]/2, params[5] + params[7]/2)

        pred_rt[t] = diffusion_trial(drift, params[4], ndt)

    return pred_rt

