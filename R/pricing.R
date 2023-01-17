# Simulation for assets, liabilities and interest rates
suppressMessages(require(yuima))
suppressMessages(require(progress))

#' simulate_num_dynamic_contagion
#'
#' @param lambda0 The initial intensity at time t = 0.
#' 
#' @param a The constant mean-reverting level.
#' 
#' @param rho The rate of arrivals for the Poisson external jumps.
#' 
#' @param delta The rate of exponential decay in intensity.
#' 
#' @param selfJumpSizeDist A function which samples intensity jump sizes for
#' self-arrivals.
#' 
#' @param extJumpSizeDist A function which samples intensity jump sizes for
#' external-arrivals.
#' 
#' @param maxTime When to stop simulating.
#' 
#' @returns The number of arrivals for a dynamic contagion process.
simulate_num_dynamic_contagion <- function(
    lambda0, a, rho, delta,
    selfJumpSizeDist, extJumpSizeDist, maxTime) {
  
  # Step 1: Set initial conditions
  prevTime <- 0
  intensity <- lambda0
  
  count <- 0
  
  while (1) {
    
    # Step 2: Simulate the next externally excited jump waiting time
    E <- -(1/rho) * log(runif(1))
    
    # Step 3: Simulate the next self-excited jump waiting time
    d <- 1 + (delta * log(runif(1))) / (intensity - a)
    
    S1 <- if (d > 0) { -(1/delta) * log(d) } else { Inf }
    S2 <- -(1/a) * log(runif(1))
    
    S <- min(S1, S2)
    
    # Step 4: Simulate the next jump time
    waitingTime <- min(S, E)
    time <- prevTime + waitingTime
    
    if (time > maxTime) {
      break
    }
    
    if (S < E) {
      count <- count + 1
    }
    
    # Step 5: Update the intensity process
    intensityPreJump <- (intensity - a) * exp(-delta*waitingTime) + a
    
    if (S < E) {
      intensity <- intensityPreJump + selfJumpSizeDist()
    } else {
      intensity <- intensityPreJump + extJumpSizeDist()
    }
    
    prevTime <- time
  }
  
  return (count)
}


#' payout_with_default
#'
#' @param V_T The value of the reinsurer's assets at maturity time T.
#' 
#' @param L_T The value of the reinsurer's liabilities at terminal time T.
#' 
#' @param C_T The value of the catastrophe losses at terminal time T.
#' 
#' @param A The attachment point specified in the reinsurance contract.
#' 
#' @param M The reinsurance cap (i.e. detachment point). 
#' 
#' @returns The payout given the final value of assets, liabilities, and
#' catastrophe losses.
payout_with_default <- function(V_T, L_T, C_T, A, M) {
  if (C_T >= M && V_T >= L_T + M - A) {
    # The reinsurance contract has hit the detachment point
    # and the reinsurer has enough assets to pay out.
    M - A
  }
  else if (C_T >= M && V_T < L_T + M - A) {
    # The reinsurance contract has hit the detachment point
    # but the reinsurer does not have enough assets to pay out the full amount.
    (V_T * (M - A) / (L_T + M - A))
  }
  else if (M > C_T && C_T >= A && V_T >= L_T + C_T - A) {
    # The reinsurance contract has not hit detachment point
    # and the reinsurer has enough assets to pay out.
    C_T - A
  }
  else if (M > C_T && C_T >= A && V_T < L_T + C_T - A) {
    # The reinsurance contract has not hit detachment point
    # but the reinsurer does not have enough assets to pay out.
    (V_T * (C_T - A) / (L_T + C_T - A))
  }
  else {
    # The catastrophe losses were not large enough to trigger the contract.
    0
  }
}


# Other parameters
maturity <- 3
markup <- 0.4

# Asset parameters
V_0 <- 130
phi_V <- -3 #* (1.3) # = V0 / L0
sigma_V <- 0.05

# Liability parameters
L_0 <- 100
phi_L <- -3
sigma_L <- 0.02

# Interest rate parameters
r_0 <- 0.02
kappa <- 0.2
m <- 0.05 # 0.5
upsil <- 0.1
lambda_r <- -0.01

# Risk neutral transformations
kappaStar <- kappa + lambda_r
mStar <- kappa * m / kappaStar

# Setup the SDEs for the (risk-neutral) assets, liabilities and interest rates
solution <- c("V", "L", "r")
drift <- c("r*V", "r*L", "kappaStar*(mStar-r)")

diffusion <- matrix(
  c(
    "phi_V*upsil*(r^{0.5})*V", "sigma_V*V", "0",
    "phi_L*upsil*(r^{0.5})*L", "0",         "sigma_L*L",
    "upsil*(r^{0.5})",         "0",         "0"
  ),
  3, 3, byrow=TRUE)

# Specify the SDE sampling grid
gridSize <- 156
Delta_t <- maturity / gridSize
grid <- setSampling(Terminal = maturity, n = gridSize)

# Create Yuima model for assets, liabilities and interest rates
true.param = list(
  kappaStar = kappaStar,
  mStar = mStar,
  upsil = upsil,
  phi_V = phi_V,
  phi_L = phi_L,
  sigma_V = sigma_V,
  sigma_L = sigma_L
)

assetAndLiabilityModel <- setModel(
  drift = drift,
  diffusion = diffusion,
  solve.variable = solution,
  xinit = c(V_0, L_0, r_0)
)

simulate_assets_and_liabilities <- function() {
  simulate(
    assetAndLiabilityModel,
    true.param = true.param,
    sampling = grid
  )
}

# Generate one sample path for illustrative purposes
set.seed(1)
X1 <- simulate_assets_and_liabilities()
plot(X1)

# Simulate many scenarios for the assets, liabilities,
# interest rates. 
simulate_market_conditions <- function(numSims) {
  pb <- progress_bar$new(total = numSims)
  
  cnames <- c()
  allPaths <- vector("list", numSims)
  
  for (i in 1:numSims) {
    pb$tick()
    
    # The interest rate simulator sometimes gives a NA value.
    # Perhaps this is because the CIR process goes negative
    # on this rough grid and that causes a crash somewhere.
    while (1) {
      path <- simulate_assets_and_liabilities()@data@original.data
      if (sum(is.na(path)) > 0) {
        # cat("NA found in simulation")
      } else {
        break
      }
    }
    
    cnames <- c(cnames, paste(c("V_", "L_", "r_"), i, sep=""))
    allPaths[[i]] <- path
  }

  allPaths <- do.call(cbind, allPaths)
  colnames(allPaths) <- cnames
  
  return(allPaths)
}


if (file.exists("allPaths.RData")) {
  # Load the simulated market conditions.
  cat("Loading simulated market scenarios from file.\n")
  load("allPaths.RData")
  numSims <- ncol(allPaths) / 3
} else {
  cat("Simulated market scenarios.\n")
  set.seed(1234)
  numSims <- 100000
  allPaths <- simulate_market_conditions(numSims)
  save(allPaths, file="allPaths.RData")
}

# Summarise the market conditions.
V_T <- rep(NA, numSims)
L_T <- rep(NA, numSims)
C_T <- rep(NA, numSims)
int_r_t <- rep(NA, numSims)

cat("Summarising the simulated market scenarios.\n")
pb <- progress_bar$new(total = numSims)

for (i in 1:numSims) {
  pb$tick()
  ind <- (i-1)*3
  V_T[i] <- allPaths[nrow(allPaths), ind+1]
  L_T[i] <- allPaths[nrow(allPaths), ind+2]
  int_r_t[i] <- sum(Delta_t * allPaths[,ind+3])
}

# Price the contract at various
# reinsurance attachment and cap levels.
calculate_prices <- function(V_T, L_T, C_T) {
  As <- c(10, 15, 20, 25, 30)
  Ms <- c(60, 65, 70, 75, 80, 85, 90)
  
  prices <- matrix(nrow=length(As), ncol=length(Ms))
  rownames(prices) <- As
  colnames(prices) <- Ms
  
  for (i in 1:length(As)) {
    for (j in 1:length(Ms)) {
      A <- As[i]
      M <- Ms[j]
      
      payouts <- rep(NA, numSims)
      payouts_with_defaults <- rep(NA, numSims)
      for (r in 1:numSims) {
        payouts[r] <- payout_with_default(V_T[r], L_T[r], C_T[r], A, M) 
      }
      discountedPayouts <- exp(-int_r_t) * payouts
      
      prices[i,j] <- (1 + markup) * mean(discountedPayouts)
    }
  }
  
  return(prices)
}


print_prices <- function(V_T, L_T, C_T) {
  digits <- 4
  price_with_default <- calculate_prices(V_T, L_T, C_T)
  print(round(price_with_default, digits))
}

price_reinsurance <- function(simulate_num_catastrophes, mu_C, sigma_C, V_T, L_T, C_T) {
  numCatastrophes <- rep(NA, numSims)
  
  set.seed(42)
  pb <- progress_bar$new(total = numSims)
  
  for (i in 1:numSims) {
    pb$tick()
    numCatastrophes[i] <- simulate_num_catastrophes()
    C_T[i] <- sum(rlnorm(numCatastrophes[i], mu_C, sigma_C))
  }
  
  cat("Mean number of catastrophes:", mean(numCatastrophes), "\n")
  
  print_prices(V_T, L_T, C_T)
  
  return(C_T)
}

# Catastrophe loss size distribution parameters
mu_C <- 2
sigma_C <- 0.5

# Simulate Poisson catastrophes.
cat("\nSimulating Poisson catastrophe process.\n")

lambda <- 0.5
simulate_poisson <- function() { rpois(1, lambda*maturity) }
C_T_pois <- price_reinsurance(simulate_poisson, mu_C, sigma_C, V_T, L_T, C_T) 

# Cox process
cat("\nSimulating Cox catastrophe process.\n")
numCatastrophes <- rep(NA, numSims)

lambda0 <- 0.49
a <- 0.4
rho <- 0.4
delta <- 1

selfJumpSizeDist <- function() { return(0) }
extJumpSizeDist <- function() { return(runif(1, 0, 0.5)) }

simulate_cox <- function() {
  simulate_num_dynamic_contagion(
    lambda0, a, rho, delta,
    selfJumpSizeDist, extJumpSizeDist, maturity
  )
}

C_T_cox <- price_reinsurance(simulate_cox, mu_C, sigma_C, V_T, L_T, C_T) 

# Hawkes process
cat("\nSimulating Hawkes catastrophe process.\n")

lambda0 <- 0.47
a <- 0.26
rho <- 0.4
delta <- 1

selfJumpSizeDist <- function() { return(runif(1, 0, 1)) }
extJumpSizeDist <- function() { return(0) }

simulate_hawkes <- function() {
  simulate_num_dynamic_contagion(
    lambda0, a, rho, delta,
    selfJumpSizeDist, extJumpSizeDist, maturity
  )
}

C_T_hawkes <- price_reinsurance(simulate_hawkes, mu_C, sigma_C, V_T, L_T, C_T) 

# Simulate dyncont catastrophes.
cat("\nSimulating dynamic contagion catastrophe process.\n")

lambda0 <- 0.29
a <- 0.26
rho <- 0.4
delta <- 1
selfJumpSizeDist <- function() { return(runif(1, 0, 1)) }
extJumpSizeDist <- function() { return(runif(1, 0, 0.5)) }

simulate_dcp <- function() {
  simulate_num_dynamic_contagion(
    lambda0, a, rho, delta,
    selfJumpSizeDist, extJumpSizeDist, maturity
  )
}

C_T_dcp <- price_reinsurance(simulate_dcp, mu_C, sigma_C, V_T, L_T, C_T) 

cat("\nCompare variance in the different catastrophe processes:\n")
cat(c( var(C_T_pois), var(C_T_cox), var(C_T_hawkes), var(C_T_dcp) ))
