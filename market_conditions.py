from pathlib import Path

import numpy as np
import numpy.random as rnd
import sdeint
from joblib import Parallel, delayed

# Create a pathlib path to the data directory (a subdirectory of this file's directory)
DATA_DIR = Path(__file__).parent / "data"


def drift_function_generator(k, eta_r, m):
    # Risk neutral transformations
    k_star = k + eta_r
    m_star = k * m / k_star

    def drift(x, t):
        V, L, r = x
        return np.array([r * V, r * L, k_star * (m_star - r)])

    return drift


def diffusion_function_generator(phi_V, sigma_V, phi_L, sigma_L, upsilon):
    def diffusion(x, t):
        V, L, r = x
        r_sqrt = np.sqrt(r)
        return np.array(
            [
                [phi_V * upsilon * r_sqrt * V, sigma_V * V, 0],
                [phi_L * upsilon * r_sqrt * L, 0, sigma_L * L],
                [upsilon * r_sqrt, 0, 0],
            ]
        )

    return diffusion


def simulate_market_conditions(
    R: int,
    seed: int,
    maturity: float,
    k: float,
    eta_r: float,
    m: float,
    phi_V: float,
    sigma_V: float,
    phi_L: float,
    sigma_L: float,
    upsilon: float,
    V_0: float,
    L_0: float,
    r_0: float,
) -> np.ndarray:
    """Simulate assets, liabilities, and interest rates for a given number of years.

    Args:
        R: The number of Monte Carlo samples to generate.
        seed: The seed for the random number generator.
        maturity: The maturity of the market in years.
        k: Mean-reversion parameter for the interest rate process.
        eta_r: The market price of interest rate risk.
        m: Long-run mean of the interest rate process.
        phi_V: Interest rate elasticity of the assets.
        sigma_V: Volatility of credit risk.
        phi_L: Interest rate elasticity of liability process.
        sigma_L: Volatility of idiosyncratic risk.
        upsilon: Volatility of the interest rate process.
        V_0: The initial value of the reinsurer's assets.
        L_0: The initial value of the reinsurer's liabilities.
        r_0: The initial value of instantaneous interest rate.

    Returns:
        A numpy array of shape (R, maturity*52, 3) representing the simulated
        asset, liability, and interest rate time series on a weekly basis.
    """

    # Setup for the SDE solving
    f = drift_function_generator(k, eta_r, m)
    G = diffusion_function_generator(phi_V, sigma_V, phi_L, sigma_L, upsilon)

    x0 = np.array([V_0, L_0, r_0])
    tspan = np.linspace(0.0, maturity, int(maturity * 52))

    # Container for all the simulated market conditions
    all_time_series = np.empty((R, len(tspan), 3), dtype=float)

    # Simulate R realisations of the SDEs.
    def simulate(seed):
        rg = rnd.default_rng(seed)

        # The interest rate simulator sometimes gives a NA value.
        # Perhaps this is because the CIR process goes negative
        # on this rough grid and that causes a crash somewhere.
        while True:
            time_series = sdeint.itoint(f, G, x0, tspan, rg)
            if np.isnan(time_series).sum() == 0:
                break
        return time_series

    delayed_simulate = delayed(simulate)

    rg = rnd.default_rng(seed)

    all_time_series = np.array(
        Parallel(n_jobs=-1)(
            delayed_simulate(seed) for seed in rg.integers(0, 2**32, size=R)
        )
    )

    return all_time_series


def get_market_conditions(
    R: int,
    seed: int,
    maturity: float,
    k: float,
    eta_r: float,
    m: float,
    phi_V: float,
    sigma_V: float,
    phi_L: float,
    sigma_L: float,
    upsilon: float,
    V_0: float,
    L_0: float,
    r_0: float,
    verbose: bool = False,
) -> np.ndarray:
    """Either loads or generates simulated assets, liabilities, and interest rates.

    Args:
        R: The number of Monte Carlo samples to generate.
        seed: The seed for the random number generator.
        maturity: The maturity of the market in years.
        k: Mean-reversion parameter for the interest rate process.
        eta_r: The market price of interest rate risk.
        m: Long-run mean of the interest rate process.
        phi_V: Interest rate elasticity of the assets.
        sigma_V: Volatility of credit risk.
        phi_L: Interest rate elasticity of liability process.
        sigma_L: Volatility of idiosyncratic risk.
        upsilon: Volatility of the interest rate process.
        V_0: The initial value of the reinsurer's assets.
        L_0: The initial value of the reinsurer's liabilities.
        r_0: The initial value of instantaneous interest rate.

    Returns:
        A numpy array of shape (R, maturity*52, 3) representing the simulated
        asset, liability, and interest rate time series on a weekly basis.
    """
    args = locals()
    args.pop("verbose")
    market_params_csv = ",".join(["{}={}".format(k, v) for k, v in args.items()])
    cache_path = DATA_DIR / f"mc-{market_params_csv}.npy"

    # Check if the array for the given input x is already cached
    if cache_path.is_file():
        # If the array is cached, load it from disk
        if verbose:
            print(f"Loading '{cache_path}'")
        all_time_series = np.load(str(cache_path))
    else:
        # If the array is not cached, create it and save it to disk
        if verbose:
            print(f"Can't find '{cache_path}', generating market conditions")
        all_time_series = simulate_market_conditions(**args)
        if verbose:
            print(f"Saving {cache_path}")
        np.save(str(cache_path), all_time_series)

    return all_time_series


def summarise_market_conditions(all_time_series, maturity):
    # Initialize variables to store the final values of assets & liabilities
    # and the integrals of the interest rate processes.
    R = all_time_series.shape[0]
    V_T = np.zeros(R)
    L_T = np.zeros(R)
    int_r_t = np.zeros(R)

    for r in range(R):
        # Store the final values of the assets and liabilities
        V_T[r] = all_time_series[r, -1, 0]
        L_T[r] = all_time_series[r, -1, 1]

        # Approximate the integral over the interest rate time series
        # using the trapezoidal rule
        int_r_t[r] = (
            np.trapz(all_time_series[r, :, 2]) / all_time_series.shape[1] * maturity
        )

    # Return the final values and integral
    return (V_T, L_T, int_r_t)
