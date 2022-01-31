data {
  int<lower=1>          N;            // number of trials
  int<lower=0, upper=1> resp[N];      // response
  real<lower=0>         rt[N];        // response time
  int<lower=1, upper=4> stim_type[N]; // stimulus type (difficulty)
}

transformed data {
  real noise[6, N]; // dynamic parameter noise (n_params, n_trials)
  
  // sample noise values
  for (i in 1:6) {
    for (n in 2:N)
      noise[i, n] = normal_rng(0, 1);
  }
}

parameters {
  real<lower=0> v[4];   // separate drift rate for each stimulus type
  real<lower=0> a;      // threshold
  real<lower=0> ndt;    // non-decision time                        
  
  real<lower=0> v_s[4]; // variation in drift rates
  real<lower=0> a_s;    // variation in threshold
  real<lower=0> ndt_s;  // variation in non-decision time 
}

transformed parameters{
  real v_t[4, N]; // trial-by-trial drift
  real a_t[N];    // trial-by-trial threshold
  real ndt_t[N];  // trial-by-trial drift
  
  // initial parameter combination
  v_t[:, 1]   = v;
  a_t[1]   = a;
  ndt_t[1] = ndt;
  
  // super statistical model
  for (i in 2:N){
    // update parameters
    v_t[1, i] = v_t[1, i - 1] + v_s[1] * noise[1, i];
    v_t[2, i] = v_t[2, i - 1] + v_s[2] * noise[2, i];
    v_t[3, i] = v_t[3, i - 1] + v_s[3] * noise[3, i];
    v_t[4, i] = v_t[4, i - 1] + v_s[4] * noise[4, i];
    
    a_t[i]    = a_t[i - 1] + a_s * noise[5, i];
    ndt_t[i]  = ndt_t[i - 1] + ndt_s * noise[6, i];
    
    // constrain parameters
    v_t[1, i] = min([max([v_t[1, i], 0.1]), 6]);
    v_t[2, i] = min([max([v_t[2, i], 0.1]), 6]);
    v_t[3, i] = min([max([v_t[3, i], 0.1]), 6]);
    v_t[4, i] = min([max([v_t[4, i], 0.1]), 6]);
    
    a_t[i]    = min([max([a_t[i], 0.3]), 2.5]);
    ndt_t[i]  = min([max([ndt_t[i], 0.1]), rt[i] - 0.001]);
  }
}

model {
  // priors
  v     ~ gamma(1.5, 1.0);
  a     ~ gamma(1.5, 1.0);
  ndt   ~ gamma(3.5, 3.0);
  
  v_s   ~ gamma(1.0, 20.0);
  a_s   ~ gamma(1.0, 20.0);
  ndt_s ~ gamma(1.0, 20.0);
  
  for (i in 1:N) {
    if (resp[i] == 1) {
      rt[i] ~ wiener(a_t[i], ndt_t[i], 0.5, v_t[stim_type[i], i]);
    } else {
        rt[i] ~ wiener(a_t[i], ndt_t[i], 0.5, -v_t[stim_type[i], i]);
    }
  }
}

generated quantities {
  real log_lik[N];
  for (i in 1:N) {
    if(resp[i]==1) {
      log_lik[i] = wiener_lpdf(rt[i] | a_t[i], ndt_t[i], 0.5, v_t[stim_type[i], i]);
    } else {
      log_lik[i] = wiener_lpdf(rt[i] | a_t[i], ndt_t[i], 0.5, -v_t[stim_type[i], i]);
    }
  }
}

