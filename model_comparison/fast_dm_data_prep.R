library(tidyverse)

# read data
df <- as.data.frame(read_csv("/Users/lukas/Documents/GitHub/dynamic_models/data/data_lexical_decision.csv"))

# write fast_dm .dat files for each person
for (sub in unique(df$id)){
  tmp <- df %>% 
    filter(id == sub) %>% 
    select(stim_type,
           resp,
           rt)
  
  path <- paste("/Users/lukas/Documents/GitHub/dynamic_models/data/fast_dm/", sub, "_respCoding.dat", sep = "")
  write_delim(tmp, path,
              delim=" ", col_names=F)
}
 

