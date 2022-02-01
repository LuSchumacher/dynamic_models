wienerProcess <- function(v=1, a=1, z=0.5, ndt=0.3){
  # Standard DDM Model Parameters
  # v   = drift rate
  # a   = boundary separation
  # z   = starting point; = 0.5 corresponds to no bias
  # ndt = non-decision time in s
  
  maxIter <- 1e4 # maximum process duration in ms
  dt <- 0.001 # time steps
  sd <- 1 # sd for noise
  sqrt_dt <- sqrt(dt*sd)
  
  # initialize diffusion path for current trial
  path <-  a * z
  # sample diffusion process noise
  noise <- rnorm(maxIter, 0, 1)
  
  # evidence accumulation process
  iter <- 1
  while (path>0 & path<a & iter<maxIter) {
    path <- path + v*dt + sqrt_dt*noise[iter]
    iter <- iter+1
  }
  
  # return choice [0, 1] and response time [s]
  return(c(as.numeric(path>a), ndt + iter*dt))
}

wienerProcess()
