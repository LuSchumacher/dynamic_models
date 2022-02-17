library(tidyverse)
library(rstan)
library(bayesplot)

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read data
df <- read_csv("../../data/data_lexical_decision.csv")

# filter for one person
df_subset <- df %>% 
  filter(id == 2)

df_subset <- df_subset %>% 
  filter(row(df_subset) <= 800)

# create stan data list
stan_data = list(
  N       = nrow(df_subset),
  correct = df_subset$acc,
  rt      = df_subset$rt,
  min_rt  = min(df_subset$rt),
  context = df_subset$stim_type)

# set initial values
init = function(chains=4) {
  L = list()
  for (c in 1:chains) {
    L[[c]]=list()
    
    L[[c]]$v     = runif(4, 0.5, 4.0)
    L[[c]]$a     = runif(1, 1.0, 1.5)
    L[[c]]$ndt   = runif(1, 0.05, 0.1)
    L[[c]]$v_s   = runif(4, 0.01, 0.05)
    L[[c]]$a_s   = runif(1, 0.01, 0.05)
    L[[c]]$ndt_S = rnorm(1, 0.01, 0.05)
    
    L[[c]]$a_t = runif(nrow(df_subset), 1.0, 1.5)
    L[[c]]$ndt_t = runif(nrow(df_subset), 0.05, 0.1)
    
  }
  return (L)
}

# set initial values
init = function(chains=4) {
  L = list()
  for (c in 1:chains) {
    L[[c]]=list()
    
    L[[c]]$v     = runif(4, 1, 1)
    L[[c]]$a     = runif(1, 1, 1)
    L[[c]]$ndt   = runif(1, 0.05, 0.05)
    L[[c]]$v_s   = runif(4, 0.001, 0.001)
    L[[c]]$a_s   = runif(1, 0.001, 0.001)
    L[[c]]$ndt_S = rnorm(1, 0.001, 0.001)
    
    L[[c]]$a_t = runif(nrow(df_subset), 1, 1)
    L[[c]]$ndt_t = runif(nrow(df_subset), 0.05, 0.05)
    
  }
  return (L)
}

fit <- stan("dynamic_ddm.stan",
            init=init(4),
            data=stan_data,
            chains=4,
            iter = 2000,
            cores=parallel::detectCores())

theta_d <- c("v[1]", "v[2]", "v[3]", "v[4]", "a", "ndt")
theta_s <- c("v_s[1]", "v_s[2]", "v_s[3]", "v_s[4]", "a_s", "ndt_s")

rstan::traceplot(fit, pars=c(theta_d, theta_s))


