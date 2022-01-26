library(tidyverse)
library(rstan)
library(bayesplot)

# set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read data
df <- read_csv("../../data/data_lexical_decision.csv")



