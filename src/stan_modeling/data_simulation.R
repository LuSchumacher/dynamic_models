library(tidyverse)
library(rstan)

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
  
  return(data_frame("context" = context,
                    "resp" = resp,
                    "rt" = rt))
}

n_obs <- 120
df <- static_diffusion_simulator(n_obs)


#------------------------------------------------------------------------#
# Model Fitting
#------------------------------------------------------------------------#



