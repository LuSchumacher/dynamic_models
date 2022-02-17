import numpy as np
from numba import njit
from tensorflow.keras.utils import to_categorical

def dynamic_prior(batch_size, fixed_var=None):
    """
    Generates a random draw from the diffusion model prior.
    """
    v = np.random.gamma(2.5, 1/1.5, (batch_size, 4))
    a = np.random.gamma(4.0, 1/3.0, batch_size)
    ndt = np.random.gamma(1.5, 1/5.0, batch_size)
    theta_s = np.random.uniform(0.0, 0.15, (batch_size, 6))
    if fixed_var != None:
        theta_s = np.random.uniform(fixed_var, fixed_var, (batch_size, 6))

    return np.c_[v, a, ndt, theta_s]


@njit
def context_gen(batch_size, n_obs):
    """
    Generates an experimenmt wiht a random sequence of 4 conditions.
    """
    obs_per_condition = int(n_obs / 4)
    context = np.zeros((batch_size, n_obs), dtype=np.int32)
    x = np.repeat([0, 1, 2, 3], obs_per_condition)
    for i in range(batch_size):
        np.random.shuffle(x)
        context[i] = x

    return context


@njit
def diffusion_trial(v, a, ndt, zr=0.5, dt=0.001, s=1.0, max_iter=1e4):
    """
    Simulates a single reaction time from a simple drift-diffusion process.
    """

    n_iter = 0
    x = a * zr
    c = np.sqrt(dt * s)
    
    while x > 0 and x < a:
        
        # DDM equation
        x += v*dt + c * np.random.randn()
        
        n_iter += 1
        
    rt = n_iter * dt
    return rt+ndt if x >= 0 else -(rt+ndt)


@njit
def dynamic_diffusion_process(prior_samples, context, n_obs):
    """
    Performs one run of a dynamic diffusion model process.
    """
    params_t, theta_s = np.split(prior_samples, 2, axis=-1)
    theta_d = np.zeros((n_obs, params_t.shape[0]))
    
    # Draw first param combination from prior
    rt = np.zeros(n_obs)
    
    # Iterate over number of trials
    for t in range(n_obs):
        
        # Run diffusion process
        rt[t] = diffusion_trial(params_t[context[t]], params_t[4], params_t[5])
        
        # Store before transition
        theta_d[t] = params_t
        
        # Transition and ensure non-negative parameters
        params_t = params_t + theta_s * np.random.randn(params_t.shape[0])
        
        # Constraints
        params_t[0] = min(max(params_t[0], 0.0), 8)
        params_t[1] = min(max(params_t[1], 0.0), 8)
        params_t[2] = min(max(params_t[2], 0.0), 8)
        params_t[3] = min(max(params_t[3], 0.0), 8)
        params_t[4] = min(max(params_t[4], 0.001), 6)
        params_t[5] = min(max(params_t[5], 0.001), 4)
        
    return np.atleast_2d(rt).T, theta_d, theta_s


@njit
def dynamic_batch_simulator(prior_samples, context):
    """
    Performs one batch of dynamic diffusion model runs.
    """
    batch_size = prior_samples.shape[0]
    n_obs = context.shape[1]
    rt = np.zeros((batch_size, n_obs, 1))
    theta_d = np.zeros((batch_size, n_obs, 6))
    theta_s = np.zeros((batch_size, 6))
    
    for i in range(batch_size):
        rt[i], theta_d[i], theta_s[i] = dynamic_diffusion_process(prior_samples[i], 
                                                                  context[i],
                                                                  n_obs)
    
    return np.concatenate((rt, np.expand_dims(context, axis=2)), axis=-1), theta_d, theta_s

@njit
def static_diffusion_process(prior_samples, context, n_obs):
    """
    Performs one run of a static diffusion model process.
    """
    
    params_t, params_stds = np.split(prior_samples, 2, axis=-1)
    
    rt = np.zeros(n_obs)
    
    # Iterate over number of trials
    for t in range(n_obs):
        
        # Run diffusion process
        rt[t] = diffusion_trial(params_t[context[t]], params_t[4], params_t[5])
        
    return np.atleast_2d(rt).T, params_t


@njit
def static_batch_simulator(prior_samples, context):
    """
    Performs one batch of static diffusion model runs.
    """
    batch_size = prior_samples.shape[0]
    n_obs = context.shape[1]
    rt = np.zeros((batch_size, n_obs, 1))
    theta = np.zeros((batch_size, n_obs, 6))

    for i in range(batch_size):
        rt[i], theta[i] = static_diffusion_process(prior_samples[i], 
                                                   context[i],
                                                   n_obs)
    return np.concatenate((rt, np.expand_dims(context, axis=2)), axis=-1), theta