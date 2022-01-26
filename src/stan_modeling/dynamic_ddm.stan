data {
  int<lower=1>          N;            // number of trials
  int<lower=0, upper=1> resp[N];      // response
  real<lower=0>         rt[N];        // response time
  int<lower=0, upper=3> stim_type[N]; // stimulus type (difficulty)
}

transformed data {
  real                  noise[6, N];  // diffusion noise (n_params, n_trials)
  
  # sample noise values
  for (i in 1:6) {
    for (n in 1:N)
      noise[i, n] = normal_rng(0, 1);
  }
}

parameters {
  real          v[size(stim_type)];   // separate drift rate for each stimulus type
  real<lower=0> a;                    // threshold
  real<lower=0> ndt;                  // non-decision time                        
  
  
  real<lower=0> v_s[size(stim_type)]; // variation in drift rates
  real<lower=0> a_s;                  // variation in threshold
  real<lower=0> ndt_s;                // variation in non-decision time 
}

transformed parameters{
  real v_t[size(stim_type)]; // trial-by-trial drift
  real a_t;   // trial-by-trial threshold
  real ndt_t; // trial-by-trial drift
  
  # super statistical model
  # initial parameter combination
  v_t   = v;
  a_t   = a;
  ndt_t = ndt;
  
  for (i in 2:N){
    v_t[1] = v_t[1] + v_s[1] * noise[1, i];
    v_t[2] = v_t[2] + v_s[2] * noise[2, i];
    v_t[3] = v_t[3] + v_s[3] * noise[3, i];
    v_t[4] = v_t[4] + v_s[4] * noise[4, i];
    
    a_t    = a_t + a_s * noise[5, i];
    ndt_t  = ndt_t + ndt_s * noise[6, i];
  }
}

model {
  # priors
  v     ~ gamma(1.5, 1.0);
  a     ~ gamma(1.5, 1.0);
  ndt   ~ gamma(3.5, 3.0);
  
  v_s   ~ gamma(1.0, 20.0);
  a_s   ~ gamma(1.0, 20.0);
  ndt_s ~ gamma(1.0, 20.0);
  
  for (i in 1:N) {
    if (resp[i] == 1) {
      rt[i] ~ wiener(a_t, ndt_t, 0.5, v_t[stim_type[i]]);
    } else {
        rt[i] ~ wiener(a_t, ndt_t, 0.5, -v_t[stim_type[i]]);
    }
  }
  
}

generated quantities {
  real log_lik[N];
  for (i in 1:N) {
    if(resp[i]==1) {
      log_lik[i] = wiener_lpdf(rt[i] | a_t, ndt_t, 0.5, v_t[stim_type[i]]);
    } else {
      log_lik[i] = wiener_lpdf(rt[i] | a_t, ndt_t, 0.5, -v_t[stim_type[i]]);
    }
  }
}

