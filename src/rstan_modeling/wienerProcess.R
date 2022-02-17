wienerProcess <- function(v, a, ndt, z=0.5, dt=0.001, s=1.0, max_iter=1e4){
  # Standard drift diffusion process
  # v        --> drift rate
  # a        --> boundary separation
  # z        --> starting point; = 0.5 corresponds to no bias
  # ndt      --> non-decision time in s
  # max_iter --> maximum process duration in ms
  # dt       --> time steps
  # sd       --> sd for noise
  
  # diffusion constant
  c <- sqrt(dt * s)
  
  # initialize diffusion path for current trial
  x <- a * z
  
  # sample diffusion process noise
  noise <- rnorm(max_iter, 0, 1)
  
  # evidence accumulation process
  n_iter <- 1
  while (x > 0 & x < a & n_iter < max_iter) {
    x <- x + v*dt + c*noise[n_iter]
    n_iter <- n_iter + 1
  }
  
  rt <- n_iter * dt
  
  return(ifelse(x >= 0, rt + ndt, -(rt + ndt)))
}