import numpy as np
from numba import njit
from generative_models import diffusion_trial

from tensorflow.keras.utils import to_categorical

@njit
def fast_dm_simulate(trials, theta, context):
    n_subs = theta.shape[0]
    pred_rt = np.empty((n_subs, trials))
    for n in range(n_subs):
        params = theta.iloc[n, 1:]

        for t in range(trials):
            drift = np.random.normal(theta[context[t]], theta[6])
            ndt = np.random.uniform(theta[5] - theta[7]/2, theta[5] + theta[7]/2)

            pred_rt[n, t] = diffusion_trial(drift, theta[4], ndt)

    return pred_rt

