from typing import Callable, Iterable, Tuple

import numpy as np
import numpy.random as rnd
from numba import njit  # type: ignore

from market_conditions import get_market_conditions, summarise_market_conditions


@njit()
def payout_without_default(C_T: np.ndarray, A: float, M: float) -> np.ndarray:
    """Calculate the payout assuming the reinsurer cannot default for catastrophe losses.

    Args:
        C_T: The value of the catastrophe losses at terminal time T.
        A: The attachment point specified in the reinsurance contract.
        M: The reinsurance cap (i.e. detachment point).

    Returns:
        The payout given the simulated catastrophe losses.
    """

    payouts = np.empty_like(C_T)

    for r in range(len(C_T)):
        if C_T[r] >= M:
            # The reinsurance contract has hit the detachment point.
            payouts[r] = M - A
        elif M > C_T[r] and C_T[r] >= A:
            # The reinsurance contract has not hit detachment point.
            payouts[r] = C_T[r] - A
        else:
            # The catastrophe losses were not large enough to trigger the contract.
            payouts[r] = 0

    return payouts


@njit()
def payout_with_default(
    V_T: np.ndarray, L_T: np.ndarray, C_T: np.ndarray, A: float, M: float
) -> np.ndarray:
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

    payouts = np.empty_like(C_T)

    for r in range(len(C_T)):
        if C_T[r] >= M and V_T[r] >= L_T[r] + M - A:
            # The reinsurance contract has hit the detachment point
            # and the reinsurer has enough assets to pay out.
            payouts[r] = M - A
        elif C_T[r] >= M and V_T[r] < L_T[r] + M - A:
            # The reinsurance contract has hit the detachment point
            # but the reinsurer does not have enough assets to pay out the full amount.
            payouts[r] = (M - A) * V_T[r] / (L_T[r] + M - A)
        elif M > C_T[r] and C_T[r] >= A and V_T[r] >= L_T[r] + C_T[r] - A:
            # The reinsurance contract has not hit detachment point
            # and the reinsurer has enough assets to pay out.
            payouts[r] = C_T[r] - A
        elif M > C_T[r] and C_T[r] >= A and V_T[r] < L_T[r] + C_T[r] - A:
            # The reinsurance contract has not hit detachment point
            # but the reinsurer does not have enough assets to pay out.
            payouts[r] = (C_T[r] - A) * V_T[r] / (L_T[r] + C_T[r] - A)
        else:
            # The catastrophe losses were not large enough to trigger the contract.
            payouts[r] = 0

    return payouts


def simulate_catastrophe_losses(
    seed: int,
    R: int,
    simulator: Callable[[int], int],
    mu_C: float,
    sigma_C: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the losses incurred by natural catastrophes.

    Args:
        seed: The seed for the random number generator.
        R: Number of Monte Carlo samples to generate.
        simulator: A function which simulates the number of catastrophes.
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
        num_catastrophes[i] = simulator(seeds[i])
        C_T[i] = np.sum(rg.lognormal(mu_C, sigma_C, size=num_catastrophes[i]))

    return C_T, num_catastrophes


def make_iterable(x):
    """Ensure that the variable can be looped over.

    Args:
        x: The variable to check.

    Returns:
        The variable if it has a length, otherwise a tuple containing the variable.
    """
    if hasattr(x, "__len__"):
        return x
    else:
        return (x,)


def calculate_prices(
    V_T: np.ndarray,
    L_T: np.ndarray,
    int_r_t: np.ndarray,
    C_T: np.ndarray,
    markup: float,
    A: float | Iterable[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
    M: float | Iterable[float] = (60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0),
) -> np.ndarray:
    """Calculate prices for reinsurance contracts at various attachment and cap levels.

    Args:
        V_T: The value of the reinsurer's assets at maturity time T.
        L_T: The value of the reinsurer's liabilities at terminal time T.
        int_r_t: The value of integral of r_t over the time interval [0, T].
        C_T: The value of the catastrophe losses at terminal time T.
        markup: The markup on the reinsurance contract.
        A: A tuple of floats containing the attachment points to consider.
        M: A tuple of floats containing the caps to consider.

    Returns:
        A numpy array of floats containing the calculated prices of reinsurance contracts.
    """
    As = make_iterable(A)
    Ms = make_iterable(M)

    prices = np.zeros((len(As), len(Ms)))
    for i in range(len(As)):
        for j in range(len(Ms)):
            payouts = payout_with_default(V_T, L_T, C_T, As[i], Ms[j])
            discounted_payouts = np.exp(-int_r_t) * payouts
            prices[i][j] = (1 + markup) * np.mean(discounted_payouts)

    return prices


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
    V_0: float | Iterable[float],
    L_0: float,
    r_0: float,
    simulator: Callable[[int], int] | Iterable[Callable[[int], int]],
    mu_C: float,
    sigma_C: float,
    markup: float,
    A: float | Iterable[float] = 20.0,
    M: float | Iterable[float] = 90.0,
    defaultable: bool = True,
    catbond: bool = False,
    K: float | Iterable[float] = 40.0,
    F: float | Iterable[float] = 10.0,
    psi_fn: Callable[
        [np.ndarray, float, float], np.ndarray
    ] = lambda C_T, K, F: np.minimum(np.maximum(C_T - K, 0), F),
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
        A: A attachment points to consider.
        M: A reinsurance caps to consider.
        defaultable: Whether or not the reinsurer can default.
        catbond: Whether or not the reinsurer has issued a catastrophe bond.
        K: The strike price of the catastrophe bond.
        F: The face value of the catastrophe bond.
        psi_fn: The catastrophe bond trigger function.


    Returns:
        A numpy array of floats containing the calculated prices of reinsurance contracts.
    """

    assert not (catbond and not defaultable), "A catbond must be defaultable."

    V_0s = make_iterable(V_0)
    simulators = make_iterable(simulator)
    As = make_iterable(A)
    Ms = make_iterable(M)
    Ks = make_iterable(K)
    Fs = make_iterable(F)

    prices = np.zeros((len(V_0s), len(simulators), len(As), len(Ms), len(Ks), len(Fs)))

    kappa = k

    for v in range(len(V_0s)):

        # If calculating the default-free price, then the initial value of the assets
        # has no effect on the price, so we can just duplicate the prices.
        if not defaultable and v > 0:
            prices[v] = prices[0]
            continue

        all_time_series = get_market_conditions(
            R,
            seed,
            maturity,
            kappa,
            eta_r,
            m,
            phi_V,
            sigma_V,
            phi_L,
            sigma_L,
            upsilon,
            V_0s[v],
            L_0,
            r_0,
        )

        V_T, L_T, int_r_t = summarise_market_conditions(all_time_series, maturity)

        for s in range(len(simulators)):

            C_T, _ = simulate_catastrophe_losses(
                seed + 1,
                R,
                simulators[s],
                mu_C,
                sigma_C,
            )

            for i in range(len(As)):
                for j in range(len(Ms)):
                    for k in range(len(Ks)):
                        for f in range(len(Fs)):
                            if catbond:
                                psi_T = psi_fn(C_T, Ks[k], Fs[f])
                                payouts = payout_with_default(
                                    V_T + psi_T,
                                    L_T,
                                    C_T,
                                    As[i],
                                    Ms[j],
                                )
                            elif defaultable:
                                payouts = payout_with_default(
                                    V_T, L_T, C_T, As[i], Ms[j]
                                )
                            else:
                                payouts = payout_without_default(C_T, As[i], Ms[j])

                            discounted_payouts = np.exp(-int_r_t) * payouts

                            prices[v, s, i, j, k, f] = (1 + markup) * np.mean(
                                discounted_payouts
                            )

    return prices.squeeze()
