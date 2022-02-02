data {
  int<lower=1>          N;            // number of trials
  int<lower=0, upper=1> resp[N];      // response
  real<lower=0>         rt[N];        // response time
  int<lower=1, upper=4> stim_type[N]; // stimulus type (difficulty)
}

parameters {
  real<lower=0.3, upper=6.0> v[2];   // separate drift rate for each stimulus type
  real<lower=0.3, upper=2.5> a;      // threshold
  real<lower=0.1, upper=1.6> ndt;    // non-decision time                        
}

model {
  // priors
  v     ~ uniform(0.3, 6.0);
  a     ~ uniform(0.3, 2.5);
  ndt   ~ uniform(0.1, 1.6);
  
  for (i in 1:N) {
      rt[i] ~ wiener(a, ndt, 0.5, v[stim_type[i]]);
      }
}

generated quantities {
  real log_lik[N];
  for (i in 1:N) {
      log_lik[i] = wiener_lpdf(rt[i] | a, ndt, 0.5, v[stim_type[i]]);
      }
}

