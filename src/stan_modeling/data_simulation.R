library(tidyverse)
library(rstan)
library(bayesplot)

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("wienerProcess2.R")

diffusion_prior <- function(){
  v1    <- runif(1, 0.3, 6.0)
  v2    <- runif(1, 0.3, 6.0)
  a     <- runif(1, 0.3, 2.5)
  ndt   <- runif(1, 0.1, 1.6)
  
  return(c(v1, v2, a, ndt))
}

generate_context <- function(n_obs){
  return(sample(1:2, n_obs, replace = T))
}


static_diffusion_simulator <- function(n_obs){
  params  <- diffusion_prior()
  context <- generate_context(n_obs)
  
  rt   <- integer(n_obs)
  resp <- integer(n_obs)
  
  for (i in 1:n_obs){
    x <- wienerProcess(params[context[i]], params[3], 0.5, params[4])
    resp[i] <- x[1]
    rt[i] <- x[2]
  }

  return(list("data" = data_frame("context" = context,
                                  "resp"    = resp,
                                  "rt"      = rt),
              "params" = params))
}

n_obs <- 120
simulation <- static_diffusion_simulator(n_obs)

df <- simulation$data
true_param <- simulation$params
#------------------------------------------------------------------------#
# Model Fitting
#------------------------------------------------------------------------#
# create stan data list
stan_data = list(
  N         = nrow(df),
  resp      = df$resp,
  rt        = df$rt,
  stim_type = df$context
)

# set initial values
init = function(chains=4) {
  L = list()
  for (c in 1:chains) {
    L[[c]]=list()
    
    L[[c]]$v   = runif(2, 0.3, 6.0)
    L[[c]]$a   = runif(1, 0.3, 2.5)
    L[[c]]$ndt = runif(1, 0.1, 0.2)
  }
  
  return (L)
}

# fit model
fit <- stan("static_ddm.stan",
            init=init(4),
            data=stan_data,
            chains=4,
            iter = 2000,
            cores=parallel::detectCores())

mcmc_pairs(fit, pars=c("v[1]", "v[2]", "a", "ndt"))
