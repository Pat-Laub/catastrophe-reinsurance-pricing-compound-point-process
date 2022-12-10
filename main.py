from pathlib import Path

import numpy as np
import numpy.random as rnd
import sdeint

from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Create a pathlib path to the data directory (a subdirectory of this file's directory)
DATA_DIR = Path(__file__).parent / "data"

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


def simulate_market_conditions(*, R, seed, maturity, kappa, lambda_r, m, phi_V, sigma_V, phi_L, sigma_L, upsilon, V_0, L_0, r_0):

    # Setup for the SDE solving
    f = drift_function_generator(kappa, lambda_r, m)
    G = diffusion_function_generator(phi_V, sigma_V, phi_L, sigma_L, upsilon)

    x0 = np.array([V_0, L_0, r_0])
    tspan = np.linspace(0.0, maturity, 156)

    rg = rnd.default_rng(seed)

    # Container for all the simulated market conditions
    all_time_series = np.empty((R, len(tspan), 3), dtype=float)

    # Simulate R realisations of the SDEs.
    def simulate(seed):
        rg = rnd.default_rng(seed)
        return sdeint.itoint(f, G, x0, tspan, rg)
    delayed_simulate = delayed(simulate)

    all_time_series = Parallel(n_jobs=-1)(delayed_simulate(seed) for seed in tqdm(rg.integers(0, 2**32, size=R)))
    
    return all_time_series

def get_market_conditions(**args):
    market_params_csv = ",".join(["{}={}".format(k, v) for k, v in args.items()])
    cache_path = DATA_DIR / f"mc-{market_params_csv}.npy"

    # Check if the array for the given input x is already cached
    if cache_path.is_file():
        # If the array is cached, load it from disk
        print(f"Loading '{cache_path}'")
        all_time_series = np.load(str(cache_path))
    else:
        # If the array is not cached, create it and save it to disk
        print(f"Can't find '{cache_path}', generating market conditions")
        all_time_series = simulate_market_conditions(**args)
        print(f"Saving {cache_path}")
        np.save(str(cache_path), all_time_series)
    
    return all_time_series


def summarise_market_conditions(all_time_series, maturity):
    # Initialize variables to store the final values of assets & liabilities
    # and the integrals of the interest rate processes.
    R = all_time_series.shape[0]
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
