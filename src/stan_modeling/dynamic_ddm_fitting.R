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
  filter(row(df_subset) <= 3200)

# create stan data list
stan_data = list(
  N         = nrow(df_subset),
  correct   = df_subset$acc,
  rt        = df_subset$rt,
  stim_type = df_subset$stim_type
)

# set initial values
init = function(chains=4) {
  L = list()
  for (c in 1:chains) {
    L[[c]]=list()
    
    L[[c]]$v     = runif(4, 0.5, 4.0)
    L[[c]]$a     = runif(1, 1.0, 1.5)
    L[[c]]$ndt   = runif(1, 0.1, 0.2)
    L[[c]]$v_s   = runif(4, 0.01, 0.05)
    L[[c]]$a_s   = runif(1, 0.01, 0.02)
    L[[c]]$ndt_S = rnorm(1, 0.01, 0.05)
  }
  return (L)
}

fit <- stan("dynamic_ddm.stan",
            init=init(4),
            data=stan_data,
            chains=4,
            iter = 2000,
            cores=parallel::detectCores())
