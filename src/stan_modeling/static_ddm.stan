data {
  int<lower=1>          N;             // number of trials
  int<lower=0, upper=1> resp[N];       // response
  real<lower=0>         rt[N];         // response time
  int<lower=1, upper=2> difficulty[N]; // stimulus difficulty
}

parameters {
  real<lower=0.0>       v[2];   // separate drift rate for each stimulus type
  real<lower=0.0>       a;      // threshold
  real<lower=0.0>       ndt;    // non-decision time                        
}

model {
  // priors
  v   ~ gamma(2.5, 1.5);
  a   ~ gamma(4, 3);
  ndt ~ gamma(1, 5);
  
  
  for (t in 1:N) {
    if (resp[t] == 1) {
      rt[t] ~ wiener(a, ndt, 0.5, v[difficulty[t]]);
    } else {
        rt[t] ~ wiener(a, ndt, 0.5, -v[difficulty[t]]);
    }
  }
}

generated quantities {
  real log_lik[N];
  for (t in 1:N) {
    if(resp[t]==1) {
      log_lik[t] = wiener_lpdf(rt[t] | a, ndt, 0.5, v[difficulty[t]]);
    } else {
      log_lik[t] = wiener_lpdf(rt[t] | a, ndt, 0.5, -v[difficulty[t]]);
    }
  }
}
