from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import sdeint

from tqdm import tqdm

# Other parameters
maturity = 3
markup = 0.4
R = int(1e5)
seed = 123

# Asset parameters
V_0 = 130
phi_V = -3  # * (1.3) # = V0 / L0
sigma_V = 0.05

# Liability parameters
L_0 = 100
phi_L = -3
sigma_L = 0.02

# Interest rate parameters
r_0 = 0.02
kappa = 0.2
m = 0.05  # 0.5
upsilon = 0.1
lambda_r = -0.01

market_params = {"R": R, "seed": seed, "kappa": kappa, "lambda_r": lambda_r, "m": m,
                 "phi_V": phi_V, "sigma_V": sigma_V, "phi_L": phi_L, "sigma_L": sigma_L,
                 "upsilon": upsilon, "V_0": V_0, "L_0": L_0, "r_0": r_0}
def drift_function_generator(kappa, lambda_r, m):
    # Risk neutral transformations
    kappa_star = kappa + lambda_r
    m_star = kappa * m / kappa_star

    def drift(x, t):
        V, L, r = x
        return np.array([r * V, r * L, kappa_star * (m_star - r)])

    return drift


def diffusion_function_generator(phi_V, sigma_V, phi_L, sigma_L, upsilon):
    def diffusion(x, t):
        V, L, r = x
        r_sqrt = np.sqrt(r)
        return np.array([
            [phi_V * upsilon * r_sqrt * V, sigma_V * V, 0],
            [phi_L * upsilon * r_sqrt * L, 0, sigma_L * L],
            [upsilon * r_sqrt, 0, 0]
        ])

    return diffusion



def simulate_market_conditions(*, R, seed, kappa, lambda_r, m, phi_V, sigma_V, phi_L, sigma_L, upsilon, V_0, L_0, r_0):

    # Setup for the SDE solving
    f = drift_function_generator(kappa, lambda_r, m)
    G = diffusion_function_generator(phi_V, sigma_V, phi_L, sigma_L, upsilon)

    x0 = np.array([V_0, L_0, r_0])
    tspan = np.linspace(0.0, maturity, 156)

    rg = rnd.default_rng(seed)

    # Container for all the simulated market conditions
    all_time_series = np.empty((R, len(tspan), 3), dtype=float)

    # Call the simulate_market_conditions function R times
    for r in tqdm(range(R)):
        # Get the time series for the current simulation
        all_time_series[r, :, :] = sdeint.itoint(f, G, x0, tspan, rg)

    return all_time_series


def summarise_market_conditions(all_time_series):
    # Initialize variables to store the final values of assets & liabilities
    # and the integrals of the interest rate processes.
    summarised_time_series = np.zeros((R, 3))

    for r in tqdm(range(R)):
        # Store the final values of the assets and liabilities
        summarised_time_series[r, 0] = all_time_series[r, -1, 0]
        summarised_time_series[r, 1] = all_time_series[r, -1, 1]

        # Approximate the integral over the interest rate time series
        # using the trapezoidal rule
        summarised_time_series[r, 2] = np.trapz(all_time_series[r, :, 2]) / all_time_series.shape[1] * maturity

    # Return the final values and integral
    return summarised_time_series


market_params_csv = ",".join(["{}={}".format(k, v) for k, v in market_params.items()])
cache_path = Path(f"market-conditions-{market_params_csv}.npy")

# Check if the array for the given input x is already cached
if cache_path.is_file():
    # If the array is cached, load it from disk
    print(f"Loading {cache_path}")
    all_time_series = np.load(str(cache_path))
else:
    # If the array is not cached, create it and save it to disk
    print(f"Can't find {cache_path}, generating market conditions")
    all_time_series = simulate_market_conditions(**market_params)
    print(f"Saving {cache_path}")
    np.save(str(cache_path), all_time_series)


summarised_time_series = summarise_market_conditions(all_time_series)

plt.hist(summarised_time_series[:, 0], 30, label="Assets")
plt.axvline(V_0, c="r", ls="--")
plt.legend()
plt.show()

plt.hist(summarised_time_series[:, 1], 30, label="Liabilities")
plt.axvline(L_0, c="r", ls="--")
plt.legend()
plt.show()

plt.hist(summarised_time_series[:, 2], 30, label="Interest rates")
plt.axvline(r_0, c="r", ls="--")
plt.legend()
plt.show()
