library(tidyverse)
library(rstan)
library(bayesplot)

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("wienerProcess2.R")

diffusion_prior <- function(){
  v     <- rgamma(2, 2.5, 1.5)
  a     <- rgamma(1, 4.0, 3.0)
  ndt   <- rgamma(1, 1.5, 5.0)
  return(c(v, a, ndt))
}


generate_context <- function(n_obs, n_condition){
  return(sample(1:n_condition, n_obs, replace = T))
}


static_diffusion_simulator <- function(n_obs, n_condition, params=NA){
  if (is.na(params[1])){
    params  <- diffusion_prior()
  }
  context <- generate_context(n_obs, n_condition)
  
  rt   <- integer(n_obs)
  resp <- integer(n_obs)
  
  for (i in 1:n_obs){
    x <- wienerProcess(params[context[i]], params[n_condition + 1], 0.5, params[n_condition + 2])
    resp[i] <- x[1]
    rt[i] <- x[2]
  }

  return(list("data" = data_frame("context" = context,
                                  "resp"    = resp,
                                  "rt"      = rt),
              "params" = params))
}
#------------------------------------------------------------------------#
# Simulation & Fitting
#------------------------------------------------------------------------#
# set initial values
init = function(chains=4) {
  L = list()
  for (c in 1:chains) {
    L[[c]]=list()
    L[[c]]$v   = runif(2, 0.5, 4.0)
    L[[c]]$a   = runif(1, 0.5, 2.5)
    L[[c]]$ndt = runif(1, 0.01, 0.01)
  }
  return (L)}

df <- tibble()
n_sim <- 100
n_obs <- 120

for (i in 1:n_sim){
  # simulate data
  simulation <- static_diffusion_simulator(n_obs, 2)
  # store data and true parameters
  pred_data <- simulation$data
  true_param <- simulation$params
  
  # create stan data list
  stan_data = list(
    N         = nrow(pred_data),
    correct   = pred_data$resp,
    rt        = pred_data$rt,
    min_rt    = min(pred_data$rt),
    stim_type = pred_data$context)
  
  # fit model
  fit <- stan("static_ddm.stan",
              init=init(4),
              data=stan_data,
              chains=4,
              iter = 4000,
              cores=parallel::detectCores())
  
  # store parameter summary stats
  tmp <- as_tibble(summary(fit, pars=c("v[1]", "v[2]", "a", "ndt"))$summary, rownames="parameter")
  # store ture parameters
  tmp <- tmp %>% 
    mutate(true_param = true_param)
  tmp$sim <- i
  
  df <- rbind(df, tmp)
}

df <- cbind(df[, 1], df[,2:ncol(df)] %>% round(digits = 3))
df$parameter[df$parameter == "v[1]"] <- "v_1"
df$parameter[df$parameter == "v[2]"] <- "v_2"

# write_csv(df, "simulation_outcome.csv")
sim_outcome <- read_csv("simulation_outcome.csv")

summary <- sim_outcome %>% 
  group_by(parameter) %>% 
  summarise(mean_sd = mean(sd),
            sd_sd = sd(sd))

#------------------------------------------------------------------------#
# PP CHECK
#------------------------------------------------------------------------#
sim_outcome <- read_csv("simulation_outcome.csv")

n_obs <- 120
simulation <- static_diffusion_simulator(n_obs, n_condition = 2, params = params)

df <- simulation$data
true_param <- simulation$params

df %>% 
  ggplot()+
  geom_density(aes(x=rt))+
  geom_density(aes(x=rt),
               data = df_subset)

mean(df$resp)
mean(df_subset$acc)
