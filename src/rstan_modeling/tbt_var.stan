data {
  int<lower=0> N;                 
  real<lower=0> rt[N];    
  int<lower=0,upper=1> correct[N];
  int<lower=1,upper=4> context[N];
}

parameters {
  real<lower=0> v[4];
  real<lower=0> a; 
  real<lower=0> ndt;
  
  real<lower=0> v_sd;
  real<lower=0> a_sd;
  real<lower=0> v_t[N];
  real<lower=0> a_t[N];
}

model {
  // Priors
  v    ~ gamma(2.5, 1.5);
  a    ~ gamma(4.0, 3.0);
  ndt  ~ gamma(1.5, 5.0);
  v_sd ~ gamma(1.5, 5.0);
  a_sd ~ gamma(1.5, 5.0);
  
  for (n in 1:N) {
     v_t[n] ~ normal(v[context[n]], v_sd)T[0, ];
     a_t[n] ~ normal(a, a_sd)T[0, ];
     if (correct[n] == 1) {
        rt[n] ~ wiener(a_t[n], ndt, 0.5, v_t[n]);
     } 
     else {
        rt[n] ~ wiener(a_t[n], ndt, 1 - 0.5, -v_t[n]);
     }
  }
}
