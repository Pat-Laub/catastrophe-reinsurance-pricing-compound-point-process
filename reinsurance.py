from typing import Callable, Tuple

import numpy as np
import numpy.random as rnd
import pandas as pd

from market_conditions import get_market_conditions, summarise_market_conditions


def payout_without_default(C_T: float, A: float, M: float) -> float:
    """Calculate the payout assuming the reinsurer cannot default for catastrophe losses.

    Args:
        C_T: The value of the catastrophe losses at terminal time T.
        A: The attachment point specified in the reinsurance contract.
        M: The reinsurance cap (i.e. detachment point).

    Returns:
        The payout given the simulated catastrophe losses.
    """

    if C_T >= M:
        # The reinsurance contract has hit the detachment point.
        return M - A
    elif M > C_T and C_T >= A:
        # The reinsurance contract has not hit detachment point.
        return C_T - A
    else:
        # The catastrophe losses were not large enough to trigger the contract.
        return 0


def payout_with_default(
    V_T: float, L_T: float, C_T: float, A: float, M: float
) -> float:
    """Calculate the payout given the final value of assets, liabilities, and catastrophe losses.

    Args:
        V_T: The value of the reinsurer's assets at maturity time T.
        L_T: The value of the reinsurer's liabilities at terminal time T.
        C_T: The value of the catastrophe losses at terminal time T.
        A: The attachment point specified in the reinsurance contract.
        M: The reinsurance cap (i.e. detachment point).

    Returns:
        The payout given the final value of assets, liabilities, and catastrophe losses.
    """

    if C_T >= M and V_T >= L_T + M - A:
        # The reinsurance contract has hit the detachment point
        # and the reinsurer has enough assets to pay out.
        return M - A
    elif C_T >= M and V_T < L_T + M - A:
        # The reinsurance contract has hit the detachment point
        # but the reinsurer does not have enough assets to pay out the full amount.
        return (M - A) * V_T / (L_T + M - A)
    elif M > C_T and C_T >= A and V_T >= L_T + C_T - A:
        # The reinsurance contract has not hit detachment point
        # and the reinsurer has enough assets to pay out.
        return C_T - A
    elif M > C_T and C_T >= A and V_T < L_T + C_T - A:
        # The reinsurance contract has not hit detachment point
        # but the reinsurer does not have enough assets to pay out.
        return (C_T - A) * V_T / (L_T + C_T - A)
    else:
        # The catastrophe losses were not large enough to trigger the contract.
        return 0


def payout_with_default_and_catbond(
    V_T: float, L_T: float, C_T: float, A: float, M: float, K: float, psi_T: float
) -> float:
    """Calculate the payout given the final value of assets, liabilities, and catastrophe losses.

    Args:
        V_T: The value of the reinsurer's assets at maturity time T.
        L_T: The value of the reinsurer's liabilities at terminal time T.
        C_T: The value of the catastrophe losses at terminal time T.
        A: The attachment point specified in the reinsurance contract.
        M: The reinsurance cap (i.e. detachment point).
        K: The catastrophe bond strike price.
        psi_T: The catastrophe bond payout.

    Returns:
        The payout given the final value of assets, liabilities, and catastrophe losses.
    """
    assert A <= K and K <= M

    if C_T >= M and V_T + psi_T >= L_T + M - A:
        # The reinsurance contract has hit the detachment point
        # and the reinsurer has enough assets (after catbond kicks in) to pay out.
        return M - A
    elif C_T >= M and V_T + psi_T < L_T + M - A:
        # The reinsurance contract has hit the detachment point but the
        # reinsurer doesn't have enough assets (even with the catbond) to pay in full.
        return (M - A) * (V_T + psi_T) / (L_T + M - A)
    elif M > C_T and C_T >= A and V_T >= L_T + C_T - A and C_T < K:
        # The reinsurance contract has not hit detachment point
        # and the reinsurer has enough assets (without catbond) to pay out.
        return C_T - A
    elif M > C_T and C_T >= A and V_T + psi_T >= L_T + C_T - A and C_T >= K:
        # The reinsurance contract has not hit detachment point
        # and the reinsurer has enough assets (with catbond) to pay out.
        return C_T - A
    elif M > C_T and C_T >= A and V_T < L_T + C_T - A and C_T < K:
        # The reinsurance contract has not hit detachment point
        # but the reinsurer does not have enough assets (without catbond) to pay out.
        return (C_T - A) * V_T / (L_T + C_T - A)
    elif M > C_T and C_T >= A and V_T + psi_T < L_T + C_T - A and C_T >= K:
        # The reinsurance contract has not hit detachment point
        # but the reinsurer does not have enough assets (even with catbond) to pay out.
        return (C_T - A) * (V_T + psi_T) / (L_T + C_T - A)
    else:
        # The catastrophe losses were not large enough to trigger the contract.
        return 0


def simulate_catastrophe_losses(
    seed: int,
    R: int,
    simulate_num_catastrophes: Callable[[int], int],
    mu_C: float,
    sigma_C: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the losses incurred by natural catastrophes.

    Args:
        seed: The seed for the random number generator.
        R: Number of Monte Carlo samples to generate.
        simulate_num_catastrophes: A function which simulates the number of catastrophes.
        mu_C: The mean of the lognormal distribution of catastrophe losses.
        sigma_C: The standard deviation of the lognormal distribution of catastrophe losses.

    Returns:
        A numpy array containing the losses incurred by natural catastrophes.
    """
    num_catastrophes = np.empty(R, dtype=int)
    rg = rnd.default_rng(seed)
    seeds = rg.integers(0, 2**32, size=R)
    C_T = np.empty(R, dtype=float)

    for i in range(R):
        num_catastrophes[i] = simulate_num_catastrophes(seeds[i])
        C_T[i] = np.sum(rg.lognormal(mu_C, sigma_C, size=num_catastrophes[i]))

    return C_T, num_catastrophes


def calculate_prices(
    V_T: np.ndarray,
    L_T: np.ndarray,
    int_r_t: np.ndarray,
    C_T: np.ndarray,
    markup: float,
    As: Tuple[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
    Ms: Tuple[float] = (60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0),
) -> pd.DataFrame | float:
    """Calculate prices for reinsurance contracts at various attachment and cap levels.

    Args:
        V_T: The value of the reinsurer's assets at maturity time T.
        L_T: The value of the reinsurer's liabilities at terminal time T.
        int_r_t: The value of integral of r_t over the time interval [0, T].
        C_T: The value of the catastrophe losses at terminal time T.
        markup: The markup on the expected value of the payout.
        As: A tuple of floats containing the attachment points to consider.
        Ms: A tuple of floats containing the caps to consider.

    Returns:
        A dataframe of floats containing the calculated prices of reinsurance contracts.
        Exceptionally, if the length of As and Ms is 1, then a float is returned.
    """
    R = len(V_T)
    prices = np.zeros((len(As), len(Ms)))
    for i in range(len(As)):
        for j in range(len(Ms)):
            A = As[i]
            M = Ms[j]

            # Create empty array to store payouts
            payouts = np.empty(R, dtype=float)
            for r in range(R):
                payouts[r] = payout_with_default(V_T[r], L_T[r], C_T[r], A, M)

            discounted_payouts = np.exp(-int_r_t) * payouts

            prices[i][j] = (1 + markup) * np.mean(discounted_payouts)

    if len(As) == 1 and len(Ms) == 1:
        return prices[0][0]

    try:
        cols = [f"$M={int(m)}$" for m in Ms]
        rows = [f"$A={int(a)}$" for a in As]
    except ValueError:
        cols = [f"$M={m}$" for m in Ms]
        rows = [f"$A={a}$" for a in As]

    df = pd.DataFrame(prices, columns=cols, index=rows)

    return df


def reinsurance_prices(
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
    V_0: float | Tuple[float],
    L_0: float,
    r_0: float,
    catastrophe_simulators: Tuple[Callable[[int], int]],
    mu_C: float,
    sigma_C: float,
    markup: float,
    As: float | Tuple[float] = 20.0,
    Ms: float | Tuple[float] = 90.0,
    defaultable: bool = True,
    catbond: bool = False,
    K: float = 0.0,
    psi_T: Callable[[float, float, float], float] = lambda C_T, K, F_cat: 0.0,
    F_cat: float = 100.0,
) -> np.ndarray:
    """Calculate reinsurance prices using Monte Carlo simulation.

    Args:
        R: The number of Monte Carlo samples to generate.
        seed: The seed for the random number generator.
        maturity: The maturity of the market in years.
        k: Mean-reversion parameter for the interest rate process.
        eta_r: The market price of interest rate risk.
        m: Long-run mean of the interest rate process.
        phi_V: Interest rate elasticity of the assets.
        sigma_V: Volatility of credit risk.
        phi_L: Interest rate elastiticity of liability process.
        sigma_L: Volatility of idiosyncratic risk.
        upsilon: Volatility of the interest rate process.
        V_0: The initial value of the reinsurer's assets.
        L_0: The initial value of the reinsurer's liabilities.
        r_0: The initial value of instantaneous interest rate.
        simulate_num_catastrophes: A function which simulates the number of natural catastrophes.
        mu_C: The mean of the lognormal distribution of catastrophe losses.
        sigma_C: The standard deviation of the lognormal distribution of catastrophe losses.
        markup: The markup on the expected value of the payout.
        As: A attachment points to consider.
        Ms: A reinsurance caps to consider.
        defaultable: Whether or not the reinsurer can default.
        catbond: Whether or not the reinsurer has issued a catastrophe bond.
        psi_T: The catastrophe bond trigger function.
        F_cat: The face value of the catastrophe bond.


    Returns:
        A dataframe of floats containing the calculated prices of reinsurance contracts.
    """

    assert not (catbond and not defaultable), "A catbond must be defaultable."

    if not hasattr(V_0, "__len__"):
        V_0 = (V_0,)
    if not hasattr(catastrophe_simulators, "__len__"):
        catastrophe_simulators = (catastrophe_simulators,)
    if not hasattr(As, "__len__"):
        As = (As,)
    if not hasattr(Ms, "__len__"):
        Ms = (Ms,)

    prices = np.zeros((len(V_0), len(catastrophe_simulators), len(As), len(Ms)))

    for v in range(len(V_0)):

        # If calculating the default-free price, then the initial value of the assets
        # has no effect on the price, so we can just duplicate the prices.
        if not defaultable and v > 0:
            prices[v] = prices[0]
            continue

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
            V_0[v],
            L_0,
            r_0,
        )

        V_T, L_T, int_r_t = summarise_market_conditions(all_time_series, maturity)

        for c in range(len(catastrophe_simulators)):

            simulate_num_catastrophes = catastrophe_simulators[c]

            C_T, _ = simulate_catastrophe_losses(
                seed + 1,
                R,
                simulate_num_catastrophes,
                mu_C,
                sigma_C,
            )

            for i in range(len(As)):
                for j in range(len(Ms)):
                    A = As[i]
                    M = Ms[j]

                    payouts = np.empty(R, dtype=float)
                    for r in range(R):
                        if catbond:
                            payouts[r] = payout_with_default_and_catbond(
                                V_T[r], L_T[r], C_T[r], A, M, K, psi_T(C_T[r], K, F_cat)
                            )
                        elif defaultable:
                            payouts[r] = payout_with_default(
                                V_T[r], L_T[r], C_T[r], A, M
                            )
                        else:
                            payouts[r] = payout_without_default(C_T[r], A, M)

                    discounted_payouts = np.exp(-int_r_t) * payouts

                    prices[v, c, i, j] = (1 + markup) * np.mean(discounted_payouts)

    prices = prices.squeeze()
    if not prices.shape:
        prices = float(prices)

    return prices


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
    V_0: float | Tuple[float],
    L_0: float,
    r_0: float,
    catastrophe_simulators: Tuple[Callable[[int], int]],
    mu_C: float,
    sigma_C: float,
    markup: float,
    As: float | Tuple[float] = 20.0,
    Ms: float | Tuple[float] = 90.0,
    K: float = 0.0,
    psi_T: Callable[[float, float, float], float] = lambda C_T, K, F_cat: 0.0,
    F_cat: float = 100.0,
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
        phi_L: Interest rate elastiticity of liability process.
        sigma_L: Volatility of idiosyncratic risk.
        upsilon: Volatility of the interest rate process.
        V_0: The initial value of the reinsurer's assets.
        L_0: The initial value of the reinsurer's liabilities.
        r_0: The initial value of instantaneous interest rate.
        simulate_num_catastrophes: A function which simulates the number of natural catastrophes.
        mu_C: The mean of the lognormal distribution of catastrophe losses.
        sigma_C: The standard deviation of the lognormal distribution of catastrophe losses.
        markup: The markup on the expected value of the payout.
        As: A attachment points to consider.
        Ms: A reinsurance caps to consider.
        defaultable: Whether or not the reinsurer can default.
        catbond: Whether or not the reinsurer has issued a catastrophe bond.
        psi_T: The catastrophe bond trigger function.
        F_cat: The face value of the catastrophe bond.


    Returns:
        A dataframe of floats containing the calculated prices of reinsurance contracts.
    """

    if not hasattr(V_0, "__len__"):
        V_0 = (V_0,)
    if not hasattr(catastrophe_simulators, "__len__"):
        catastrophe_simulators = (catastrophe_simulators,)
    if not hasattr(As, "__len__"):
        As = (As,)
    if not hasattr(Ms, "__len__"):
        Ms = (Ms,)

    assert (
        len(V_0) == 1
    ), "Doesn't change the price if there are differing initial values for assets."

    prices = np.zeros((len(V_0), len(catastrophe_simulators), len(As), len(Ms)))

    for v in range(len(V_0)):

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
            V_0[v],
            L_0,
            r_0,
        )

        V_T, L_T, int_r_t = summarise_market_conditions(all_time_series, maturity)

        discounted_bond_payouts = np.mean(np.exp(-int_r_t) * F_cat)

        for c in range(len(catastrophe_simulators)):

            simulate_num_catastrophes = catastrophe_simulators[c]

            C_T, _ = simulate_catastrophe_losses(
                seed + 1,
                R,
                simulate_num_catastrophes,
                mu_C,
                sigma_C,
            )

            for i in range(len(As)):
                for j in range(len(Ms)):
                    A = As[i]
                    M = Ms[j]

                    catbond_payouts = np.empty(R, dtype=float)
                    for r in range(R):
                        catbond_payouts[r] = psi_T(C_T[r], K, F_cat)

                    discounted_catbond_payouts = np.exp(-int_r_t) * catbond_payouts

                    delta_0 = np.mean(discounted_bond_payouts) - np.mean(
                        discounted_catbond_payouts
                    )
                    prices[v, c, i, j] = (1 + markup) * delta_0

    prices = prices.squeeze()
    if not prices.shape:
        prices = float(prices)

    return prices
