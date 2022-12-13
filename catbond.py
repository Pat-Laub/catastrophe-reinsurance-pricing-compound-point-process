from typing import Callable, Tuple

import numpy as np

from market_conditions import *
from reinsurance import *


def catbond_prices(
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
    simulator: Callable[[int], int] | Tuple[Callable[[int], int], ...],
    mu_C: float,
    sigma_C: float,
    markup: float,
    K: float = 0.0,
    F: float = 100.0,
    psi_fn: Callable[
        [np.ndarray, float, float], np.ndarray
    ] = lambda C_T, K, F: np.minimum(np.maximum(C_T - K, 0), F),
) -> np.ndarray:
    """Calculate catbond prices using Monte Carlo simulation.

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
        simulator: A function which simulates the number of natural catastrophes.
        mu_C: The mean of the lognormal distribution of catastrophe losses.
        sigma_C: The standard deviation of the lognormal distribution of catastrophe losses.
        markup: The markup on the reinsurance contract.
        K: The strike price of the catastrophe bond.
        F: The face value of the catastrophe bond.
        psi_fn: The catastrophe bond trigger function.


    Returns:
        A numpy array containing the calculated prices of the catastrophe bonds.
    """

    simulators = make_iterable(simulator)
    Ks = make_iterable(K)
    Fs = make_iterable(F)

    prices = np.zeros((len(simulators), len(Ks), len(Fs)))

    all_time_series = get_market_conditions(
        R,
        seed,
        maturity,
        k,
        eta_r,
        m,
        phi_V,
        sigma_V,
        phi_L,
        sigma_L,
        upsilon,
        V_0,
        L_0,
        r_0,
    )

    _, _, int_r_t = summarise_market_conditions(all_time_series, maturity)

    discounted_bond_payouts = np.mean(np.exp(-int_r_t) * F)

    for s in range(len(simulators)):

        C_T, _ = simulate_catastrophe_losses(
            seed + 1,
            R,
            simulators[s],
            mu_C,
            sigma_C,
        )

        catbond_payouts = F - psi_fn(C_T, K, F)
        discounted_catbond_payouts = np.exp(-int_r_t) * catbond_payouts
        delta_0 = np.mean(discounted_bond_payouts) - np.mean(discounted_catbond_payouts)

        prices[s] = (1 + markup) * delta_0

    return prices.squeeze()


def net_present_value(
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
    simulator: Callable[[int], int] | Tuple[Callable[[int], int], ...],
    mu_C: float,
    sigma_C: float,
    markup: float,
    catbond_markup: float,
    K: float = 0.0,
    F: float = 100.0,
    psi_fn: Callable[
        [np.ndarray, float, float], np.ndarray
    ] = lambda C_T, K, F: np.minimum(np.maximum(C_T - K, 0), F),
) -> np.ndarray:
    """The reinsurer's net present value.

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
        simulator: A function which simulates the number of natural catastrophes.
        mu_C: The mean of the lognormal distribution of catastrophe losses.
        sigma_C: The standard deviation of the lognormal distribution of catastrophe losses.
        markup: The markup on the reinsurance contract.
        catbond_markup: The markup on the catbond.
        K: The catastrophe bond strike price.
        F: The face value of the catastrophe bond.
        psi_fn: The catastrophe bond trigger function.


    Returns:
        An array of net present values for the reinsurer.
    """

    simulators = make_iterable(simulator)

    npvs = np.zeros((len(simulators)))

    for s in range(len(simulators)):
        reinsurance_pv = reinsurance_prices(
            R,
            seed,
            maturity,
            k,
            eta_r,
            m,
            phi_V,
            sigma_V,
            phi_L,
            sigma_L,
            upsilon,
            V_0,
            L_0,
            r_0,
            simulators[s],
            mu_C,
            sigma_C,
            markup=0.0,
            catbond=True,
            K=K,
            F=F,
        )

        delta_0 = catbond_prices(
            R,
            seed,
            maturity,
            k,
            eta_r,
            m,
            phi_V,
            sigma_V,
            phi_L,
            sigma_L,
            upsilon,
            V_0,
            L_0,
            r_0,
            simulators[s],
            mu_C,
            sigma_C,
            markup=0.0,
            K=K,
            F=F,
            psi_fn=psi_fn,
        )

        npvs[s] = markup * reinsurance_pv - catbond_markup * delta_0

    return npvs.squeeze()


def total_hedging_cost(
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
    simulator: Callable[[int], int] | Tuple[Callable[[int], int], ...],
    mu_C: float,
    sigma_C: float,
    markup: float,
    catbond_markup: float,
    K_ins: float = 0.0,
    F_ins: float = 100.0,
    K_reins: float = 0.0,
    F_reins: float = 100.0,
    psi_fn: Callable[
        [np.ndarray, float, float], np.ndarray
    ] = lambda C_T, K, F: np.minimum(np.maximum(C_T - K, 0), F),
) -> np.ndarray:
    """The insurer's total hedging cost (the sum of the reinsurance price and the catbond price).

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
        simulator: A function which simulates the number of natural catastrophes.
        mu_C: The mean of the lognormal distribution of catastrophe losses.
        sigma_C: The standard deviation of the lognormal distribution of catastrophe losses.
        markup: The markup on the reinsurance contract.
        catbond_markup: The markup on the catbond.
        K_ins: float = 0.0,
        F_ins: float = 100.0,
        K_reins: float = 0.0,
        F_reins: float = 100.0,
        psi_fn: The catastrophe bond trigger function.


    Returns:
        A numpy array of the insurer's total hedging costs.
    """

    reinsurance_pv = reinsurance_prices(
        R,
        seed,
        maturity,
        k,
        eta_r,
        m,
        phi_V,
        sigma_V,
        phi_L,
        sigma_L,
        upsilon,
        V_0,
        L_0,
        r_0,
        simulator,
        mu_C,
        sigma_C,
        markup=0.0,
        catbond=True,
        K=K_reins,
        F=F_reins,
        psi_fn=psi_fn,
    )

    delta_ins = catbond_prices(
        R,
        seed,
        maturity,
        k,
        eta_r,
        m,
        phi_V,
        sigma_V,
        phi_L,
        sigma_L,
        upsilon,
        V_0,
        L_0,
        r_0,
        simulator,
        mu_C,
        sigma_C,
        markup=0.0,
        K=K_ins,
        F=F_ins,
        psi_fn=psi_fn,
    )

    return markup * reinsurance_pv + catbond_markup * delta_ins
