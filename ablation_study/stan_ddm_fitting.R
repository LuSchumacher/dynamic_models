library(tidyverse)
library(rstan)
library(reticulate)
library(bayesplot)
pd <- import("pandas")

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read sim data
sim_data_noVar <- pd$read_pickle("sim_data_ablation/sim_data_ablation_3200_noVar.pkl")
sim_data_lowVar <- pd$read_pickle("sim_data_ablation/sim_data_ablation_3200_lowVar.pkl")
sim_data_highVar <- pd$read_pickle("sim_data_ablation/sim_data_ablation_3200_highVar.pkl")

ablation_fitting <- function(data) {
  # initialize data structure
  post_samples <- read.csv("sim_data_ablation/stan_post_noVar.csv")
  for (sim in 10:100) {
    # subset data
    rt <- abs(sim_data_noVar$rt[sim, ])
    context <- sim_data_noVar$context[sim, ] + 1
    correct <- ifelse(sim_data_noVar$rt[sim, ] >= 0, 1, 0)
    rt_min <- min(rt)
    
    # create stan data list
    stan_data = list(
      N       = length(rt),
      correct = correct,
      rt      = rt,
      context = context
    )
    
    # set initial values
    init = function(chains=4) {
      L = list()
      for (c in 1:chains) {
        L[[c]]=list()
        L[[c]]$v     = rgamma(4, 2.5, 1.5)
        L[[c]]$a     = rgamma(1, 4.0, 3.0)
        L[[c]]$ndt   = rt_min * 0.9
      }
      return (L)
    }
    
    # fit model
    fit <- rstan::stan("static_ddm.stan",
                       init=init(4),
                       data=stan_data,
                       chains=4,
                       iter=2000,
                       cores=parallel::detectCores(),
                       control=list(adapt_delta=0.99,
                                    max_treedepth=15))
    
    # store data
    tmp <- as.data.frame(rstan::extract(fit))
    tmp$sim <- sim
    post_samples <- rbind(post_samples, tmp)
    write.csv(post_samples, "sim_data_ablation/stan_post_noVar.csv", row.names = F)
    print(paste("Simulation:", sim, "is finished..."))
  }
  
  return(post_samples)
}

post_noVar <- ablation_fitting(sim_data_noVar)



