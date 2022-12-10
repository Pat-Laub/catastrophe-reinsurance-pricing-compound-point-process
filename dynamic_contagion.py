import numpy as np
from typing import Callable, Type


def simulate_num_dynamic_contagion(
    rg: Type[np.random.Generator],
    maxTime: float,
    lambda0: float,
    a: float,
    rho: float,
    delta: float,
    selfJumpSizeDist: Callable[[], float],
    extJumpSizeDist: Callable[[], float],
) -> int:
    """
    Simulate a dynamic contagion process and return the number of arrivals.

    :param rg: A random number generator.
    :param maxTime: When to stop simulating.
    :param lambda0: The initial intensity at time t = 0.
    :param a: The constant mean-reverting level.
    :param rho: The rate of arrivals for the Poisson external jumps.
    :param delta: The rate of exponential decay in intensity.
    :param selfJumpSizeDist: A function which samples intensity jump sizes for self-arrivals.
    :param extJumpSizeDist: A function which samples intensity jump sizes for external-arrivals.

    :returns: The number of arrivals for a dynamic contagion process.
    """

    # Step 1: Set initial conditions
    prevTime = 0
    intensity = lambda0

    count = 0

    while True:
        # Step 2: Simulate the next externally excited jump waiting time
        E: float = (1 / rho) * rg.exponential()

        # Step 3: Simulate the next self-excited jump waiting time
        d: float = 1 - (delta * rg.exponential()) / (intensity - a)

        S1: float = -(1 / delta) * np.log(d) if d > 0 else float("inf")
        S2: float = (1 / a) * rg.exponential()

        S = min(S1, S2)

        # Step 4: Simulate the next jump time
        waitingTime = min(S, E)
        assert waitingTime > 0

        time = prevTime + waitingTime

        if time > maxTime:
            break

        if S < E:
            count += 1

        # Step 5: Update the intensity process
        intensityPreJump: float = (intensity - a) * np.exp(-delta * waitingTime) + a

        if S < E:
            intensity = intensityPreJump + selfJumpSizeDist(rg)
        else:
            intensity = intensityPreJump + extJumpSizeDist(rg)

        prevTime = time

    return count
