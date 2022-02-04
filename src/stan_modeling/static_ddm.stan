data {
  int<lower=1>          N;            // number of trials
  int<lower=0, upper=1> correct[N];   // correctness of response
  real<lower=0>         rt[N];        // response time
  real<lower=0>         min_rt;       // smallest response time in data
  int<lower=1, upper=2> stim_type[N]; // stimulus type
}

parameters {
  real<lower=0.0>               v[2];   // separate drift rate for each stimulus type
  real<lower=0.0>               a;      // threshold
  real<lower=0.0, upper=min_rt> ndt;    // non-decision time                        
}

model {
  // priors
  v   ~ gamma(2.5, 1.5);
  a   ~ gamma(4.0, 3.0);
  ndt ~ gamma(1.5, 5.0);
  
  for (t in 1:N) {
    if (correct[t] == 1) {
      rt[t] ~ wiener(a, ndt, 0.5, v[stim_type[t]]);
    } else {
        rt[t] ~ wiener(a, ndt, 0.5, -v[stim_type[t]]);
    }
  }
}

generated quantities {
  real log_lik[N];
  for (t in 1:N) {
    if(correct[t]==1) {
      log_lik[t] = wiener_lpdf(rt[t] | a, ndt, 0.5, v[stim_type[t]]);
    } else {
      log_lik[t] = wiener_lpdf(rt[t] | a, ndt, 0.5, -v[stim_type[t]]);
    }
  }
}
