import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Callable


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

    if (C_T >= M) and (V_T >= L_T + M - A):
        # The reinsurance contract has hit the detachment point
        # and the reinsurer has enough assets to pay out.
        return M - A
    elif (C_T >= M) and (V_T < L_T + M - A):
        # The reinsurance contract has hit the detachment point
        # but the reinsurer does not have enough assets to pay out the full amount.
        return V_T * (M - A) / (L_T + M - A)
    elif (M > C_T) and (C_T >= A) and (V_T >= L_T + C_T - A):
        # The reinsurance contract has not hit detachment point
        # and the reinsurer has enough assets to pay out.
        return C_T - A
    elif (M > C_T) and (C_T >= A) and (V_T < L_T + C_T - A):
        # The reinsurance contract has not hit detachment point
        # but the reinsurer does not have enough assets to pay out.
        return V_T * (C_T - A) / (L_T + C_T - A)
    else:
        # The catastrophe losses were not large enough to trigger the contract.
        return 0


def simulate_catastrophe_losses(
    rg: np.random.Generator,
    R: int,
    simulate_num_catastrophes: Callable[[np.random.Generator], int],
    mu_C: float,
    sigma_C: float,
) -> np.array:
    """Simulate the losses incurred by natural catastrophes.

    Args:
        rg: An instance of the numpy.random.Generator class.
        R: Number of Monte Carlo samples to generate.
        simulate_num_catastrophes: A function which simulates the number of catastrophes.
        mu_C: The mean of the lognormal distribution of catastrophe losses.
        sigma_C: The standard deviation of the lognormal distribution of catastrophe losses.

    Returns:
        A numpy array containing the losses incurred by natural catastrophes.
    """
    num_catastrophes = np.empty(R, dtype=int)
    C_T = np.empty(R)

    for i in tqdm(range(R)):
        num_catastrophes[i] = simulate_num_catastrophes(rg)
        C_T[i] = np.sum(rg.lognormal(mu_C, sigma_C, size=num_catastrophes[i]))

    return C_T, num_catastrophes


def calculate_prices(
    V_T: np.ndarray,
    L_T: np.ndarray,
    int_r_t: np.ndarray,
    C_T: np.ndarray,
    markup: float,
) -> np.ndarray:
    """Calculate prices for reinsurance contracts at various attachment and cap levels.

    Args:
        V_T: The value of the reinsurer's assets at maturity time T.
        L_T: The value of the reinsurer's liabilities at terminal time T.
        int_r_t: The value of integral of r_t over the time interval [0, T].
        C_T: The value of the catastrophe losses at terminal time T.
        markup: The markup on the expected value of the payout.

    Returns:
        A numpy array of floats containing the calculated prices of reinsurance contracts.
    """
    As = [10, 15, 20, 25, 30]
    Ms = [60, 65, 70, 75, 80, 85, 90]

    prices = np.zeros((len(As), len(Ms)))
    for i in range(len(As)):
        for j in range(len(Ms)):
            A = As[i]
            M = Ms[j]

            # Create empty array to store payouts
            payouts = np.empty(len(V_T))
            for k in range(len(V_T)):
                payouts[k] = payout_with_default(V_T[k], L_T[k], C_T[k], A, M)

            discounted_payouts = np.exp(-int_r_t) * payouts

            prices[i][j] = (1 + markup) * np.mean(discounted_payouts)

    df = pd.DataFrame(
        prices, columns=[f"M={m}" for m in Ms], index=[f"A={a}" for a in As]
    )

    return df
