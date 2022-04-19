library(tidyverse)
library(rstan)
library(bayesplot)

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("wienerProcess.R")

diffusion_prior <- function(n_condition=4){
  v     <- rgamma(n_condition, 2.5, 1.5)
  a     <- rgamma(1, 4.0, 3.0)
  ndt   <- rgamma(1, 1.5, 5.0)
  return(c(v, a, ndt))
}

generate_context <- function(n_obs, n_condition=4){
  obs_per_condition <- as.integer(n_obs / n_condition)
  conditions <- 1:n_condition
  context <- rep(conditions, obs_per_condition)
  context <- sample(context, replace = F)
  return(context)
}

static_diffusion_simulator <- function(n_obs, n_condition, params=NA){
  if (is.na(params[1])){
    params  <- diffusion_prior()
  }
  
  context <- generate_context(n_obs, n_condition)
  
  rt   <- integer(n_obs)
  resp <- integer(n_obs)
  
  for (i in 1:n_obs){
    rt[i] <- wienerProcess(v=params[context[i]], a=params[n_condition + 1], ndt=params[n_condition + 2])
    resp[i] <- ifelse(rt[i] >= 0, 1, 0)
  }
  
  return(list("data" = data_frame("context" = context,
                                  "resp"    = resp,
                                  "rt"      = rt,
                                  "rt_abs"  = abs(rt)),
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
    L[[c]]$v   = runif(4, 0.5, 4.0)
    L[[c]]$a   = runif(1, 0.5, 2.5)
    L[[c]]$ndt = runif(1, 0.01, 0.01)
  }
  return (L)}

summary_df <- tibble()
sim_data <- tibble()
n_sim <- 1
n_obs <- 3200

for (i in 1:n_sim){
  # simulate data
  simulation <- static_diffusion_simulator(n_obs, 4)
  # store data and true parameters
  tmp_sim_data <- simulation$data
  true_param <- simulation$params
  
  # create stan data list
  stan_data = list(
    N       = nrow(tmp_sim_data),
    correct = tmp_sim_data$resp,
    rt      = tmp_sim_data$rt_abs,
    min_rt  = min(tmp_sim_data$rt_abs),
    context = tmp_sim_data$context)
  
  # fit model
  fit <- stan("static_ddm.stan",
              init=init(4),
              data=stan_data,
              chains=4,
              iter = 2000,
              cores=parallel::detectCores())
  
  # store parameter summary stats
  tmp_summary <- as_tibble(summary(fit, pars=c("v[1]", "v[2]", "v[3]", "v[4]", "a", "ndt"))$summary, rownames="parameter")
  # store true parameters
  tmp_summary <- tmp_summary %>% 
    mutate(true_param = true_param)
  
  tmp_summary$sim <- i
  tmp_sim_data$sim <- i
  
  summary_df <- rbind(summary_df, tmp_summary)
  sim_data <- rbind(sim_data, tmp_sim_data)
}

summary_df <- cbind(summary_df[, 1], summary_df[,2:ncol(summary_df)] %>% round(digits = 3))
summary_df$parameter[summary_df$parameter == "v[1]"] <- "v_1"
summary_df$parameter[summary_df$parameter == "v[2]"] <- "v_2"

write_csv(summary_df, "simulation_outcome.csv")
write_csv(sim_data, "sim_data.csv")
# sim_outcome <- read_csv("simulation_summary.csv")
# sim_data <- read_csv("sim_data.csv")

summary <- summary_df %>% 
  group_by(parameter) %>% 
  summarise(mean_sd = mean(sd),
            sd_sd = sd(sd))




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

# fit model
fit <- stan("static_ddm.stan",
            init=init(4),
            data=stan_data,
            chains=4,
            iter = 2000,
            cores=parallel::detectCores())
