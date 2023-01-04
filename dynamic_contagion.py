from typing import Callable

import numpy as np
import numpy.random as rnd
from numba import njit  # type: ignore


def simulate_num_dynamic_contagion(
    rg: np.random.Generator,
    max_time: float,
    lambda0: float,
    a: float,
    rho: float,
    delta: float,
    self_jump_size_dist: Callable[[np.random.Generator], float],
    ext_jump_size_dist: Callable[[np.random.Generator], float],
) -> int:
    """Simulate a dynamic contagion process and return the number of arrivals.

    Args:
        rg: A random number generator.
        max_time: When to stop simulating.
        lambda0: The initial intensity at time t = 0.
        a: The constant mean-reverting level.
        rho: The rate of arrivals for the Poisson external jumps.
        delta: The rate of exponential decay in intensity.
        self_jump_size_dist: A function which samples intensity jump sizes for self-arrivals.
        ext_jump_size_dist: A function which samples intensity jump sizes for external-arrivals.

    Returns:
        The number of arrivals for a dynamic contagion process.
    """

    # Step 1: Set initial conditions
    prev_time = 0.0
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
        waiting_time = min(S, E)
        assert waiting_time > 0

        time = prev_time + waiting_time

        if time > max_time:
            break

        if S < E:
            count += 1

        # Step 5: Update the intensity process
        intensity_pre_jump: float = (intensity - a) * np.exp(-delta * waiting_time) + a

        if S < E:
            intensity = intensity_pre_jump + self_jump_size_dist(rg)
        else:
            intensity = intensity_pre_jump + ext_jump_size_dist(rg)

        prev_time = time

    return count


@njit()
def simulate_num_dynamic_contagion_uniform_jumps(
    seed: int,
    max_time: float,
    lambda0: float,
    a: float,
    rho: float,
    delta: float,
    self_jump_min: float,
    self_jump_max: float,
    ext_jump_min: float,
    ext_jump_max: float,
) -> int:
    """Simulate a dynamic contagion process and return the number of arrivals.

    Args:
        seed: The seed for the random number generator.
        max_time: When to stop simulating.
        lambda0: The initial intensity at time t = 0.
        a: The constant mean-reverting level.
        rho: The rate of arrivals for the Poisson external jumps.
        delta: The rate of exponential decay in intensity.
        self_jump_min: The minimum jump size for self-arrivals.
        self_jump_max: The maximum jump size for self-arrivals.
        ext_jump_min: The minimum jump size for external-arrivals.
        ext_jump_max: The maximum jump size for external-arrivals.

    Returns:
        The number of arrivals for a dynamic contagion process.
    """
    rnd.seed(seed)

    # Step 1: Set initial conditions
    prev_time = 0.0
    intensity = lambda0

    count = 0

    while True:
        # Step 2: Simulate the next externally excited jump waiting time
        E: float = (1 / rho) * rnd.exponential()

        # Step 3: Simulate the next self-excited jump waiting time
        d: float = 1 - (delta * rnd.exponential()) / (intensity - a)

        S2: float = (1 / a) * rnd.exponential()

        if d > 0:
            S1: float = -(1 / delta) * np.log(d)
            S = min(S1, S2)
        else:
            S = S2

        # Step 4: Simulate the next jump time
        waiting_time = min(S, E)
        assert waiting_time > 0

        time = prev_time + waiting_time

        if time > max_time:
            break

        if S < E:
            count += 1

        # Step 5: Update the intensity process
        intensity_pre_jump: float = (intensity - a) * np.exp(-delta * waiting_time) + a

        if S < E:
            self_jump_size = rnd.uniform(self_jump_min, self_jump_max)
            intensity = intensity_pre_jump + self_jump_size
        else:
            ext_jump_size = rnd.uniform(ext_jump_min, ext_jump_max)
            intensity = intensity_pre_jump + ext_jump_size

        prev_time = time

    return count
