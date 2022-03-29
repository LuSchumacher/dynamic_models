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

# create Stan data list
stan_data = list(
  N       = nrow(df_subset),
  correct = df_subset$acc,
  rt      = df_subset$rt,
  # min_rt  = min(df_subset$rt),
  context = df_subset$stim_type)

# set initial values
init = function(chains=4) {
  L = list()
  for (c in 1:chains) {
    L[[c]]=list()
    
    L[[c]]$v     = runif(4, 0.5, 4.0)
    L[[c]]$a     = runif(1, 1.0, 1.5)
    L[[c]]$ndt   = runif(1, 0.05, 0.1)
    L[[c]]$v_sd  = runif(1, 0.01, 0.05)
    L[[c]]$a_sd  = runif(1, 0.01, 0.05)
    # L[[c]]$v_t   = runif(nrow(df_subset), 1.0, 3.0)
    # L[[c]]$a_t   = runif(nrow(df_subset), 1.3, 1.7)
  }
  return (L)
}

fit <- stan("tbt_var.stan",
            init=init(4),
            data=stan_data,
            chains=4,
            iter = 4000,
            cores=parallel::detectCores())

mcmc_pairs(fit, pars = c("v[1]", "v[2]", "v[3]", "v[4]", "a", "ndt", "v_sd", "a_sd"))

fit


##------------------------------------------------------------------------------------##
## StanDDM testing
##------------------------------------------------------------------------------------##
devtools::install_github('https://github.com/Seneketh/StanDDM.git', ref = 'master')
library(StanDDM)
library(tidyverse)
library(magrittr)
library(rstan)


# data prep
stanDDM_data <- df_subset %>% 
  select(id, rt, stim_type, acc) %>% 
  rename(suj=id,
         crit=stim_type,
         cor=acc)

stanDDM_data <- experimental_data_processing(stanDDM_data)

fit <- StanDDM::StanDDM(data = stanDDM_data,
                 simulation = FALSE,
                 include_models = 'st_sv')



df %>% 
  mutate(id = replace(...)) %>% 
  mutate(id = str_replace(...))


