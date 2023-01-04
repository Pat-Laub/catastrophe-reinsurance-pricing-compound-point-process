# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2
#   kernelspec:
#     display_name: Python [conda env:catbond]
#     language: python
#     name: conda-env-catbond-py
# ---

# +
import matplotlib.pyplot as plt

import __module_import__
from dynamic_contagion import *
# -

maturity = 3
R = 1_000_000


# +
# Poisson process
def simulate_poisson(rg):
    lambda_ = 0.5
    return rg.poisson(lambda_ * maturity)


rg = np.random.default_rng(123)
N_T = [simulate_poisson(rg) for _ in range(R)]
np.mean(N_T), np.var(N_T)


# +
# %%time
# Cox proces
def simulate_cox_slow(rg):
    lambda0 = 0.49
    a = 0.4
    rho = 0.4
    delta = 1

    selfJumpSizeDist = lambda rg: 0
    extJumpSizeDist = lambda rg: rg.uniform(0, 0.5)

    return simulate_num_dynamic_contagion(rg, maturity, lambda0, a, rho, delta, selfJumpSizeDist, extJumpSizeDist)


seed = 123
rg = np.random.default_rng(seed)
N_T = [simulate_cox_slow(rg) for _ in range(R)]
np.mean(N_T), np.var(N_T)


# +
# %%time
# Cox proces
def simulate_cox(rg):
    lambda0 = 0.49
    a = 0.4
    rho = 0.4
    delta = 1

    return simulate_num_dynamic_contagion_uniform_jumps(
        rg.integers(0, 2**32), maturity, lambda0, a, rho, delta, 0, 0, 0, 0.5
    )


rg = np.random.default_rng()
N_T = [simulate_cox(rg) for _ in range(R)]
np.mean(N_T), np.var(N_T)


# +
# %%time
# Hawkes process
def simulate_hawkes_slow(rg):
    lambda0 = 0.47
    a = 0.26
    rho = 0.4
    delta = 1

    selfJumpSizeDist = lambda rg: rg.uniform()
    extJumpSizeDist = lambda rg: 0

    return simulate_num_dynamic_contagion(rg, maturity, lambda0, a, rho, delta, selfJumpSizeDist, extJumpSizeDist)


rg = np.random.default_rng(123)
N_T = [simulate_hawkes_slow(rg) for _ in range(R)]
np.mean(N_T), np.var(N_T)


# +
# %%time
# Hawkes process
def simulate_hawkes(rg):
    lambda0 = 0.47
    a = 0.26
    rho = 0.4
    delta = 1

    return simulate_num_dynamic_contagion_uniform_jumps(
        rg.integers(0, 2**32), maturity, lambda0, a, rho, delta, 0.0, 1.0, 0.0, 0.0
    )


rg = np.random.default_rng(123)
N_T = [simulate_hawkes(rg) for _ in range(R)]
np.mean(N_T), np.var(N_T)


# +
# %%time
# Dynamic contagion process
def simulate_dcp_slow(rg):
    lambda0 = 0.29
    a = 0.26
    rho = 0.4
    delta = 1

    selfJumpSizeDist = lambda rg: rg.uniform()
    extJumpSizeDist = lambda rg: rg.uniform(0, 0.5)

    return simulate_num_dynamic_contagion(rg, maturity, lambda0, a, rho, delta, selfJumpSizeDist, extJumpSizeDist)


rg = np.random.default_rng(123)
N_T = [simulate_dcp_slow(rg) for _ in range(R)]
np.mean(N_T), np.var(N_T)


# +
# %%time
# Dynamic contagion process
def simulate_dcp(rg):
    lambda0 = 0.29
    a = 0.26
    rho = 0.4
    delta = 1

    return simulate_num_dynamic_contagion_uniform_jumps(
        rg.integers(0, 2**32), maturity, lambda0, a, rho, delta, 0.0, 1.0, 0.0, 0.5
    )


rg = np.random.default_rng(123)
N_T = [simulate_dcp(rg) for _ in range(R)]
np.mean(N_T), np.var(N_T)
