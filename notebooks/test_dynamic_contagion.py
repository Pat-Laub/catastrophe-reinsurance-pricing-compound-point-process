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
#     display_name: catbond
#     language: python
#     name: python3
# ---

# +
# Add the parent directory to the path so that we can import the modules.
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from dynamic_contagion import *
import matplotlib.pyplot as plt
# -

maturity = 3
R = 1_000_000

# +
# Poisson process
print("\nSimulating Poisson catastrophe process.\n")

lambda_ = 0.5

def simulate_poisson(rg):
    return rg.poisson(lambda_ * maturity)

rg = np.random.default_rng(123)
np.mean([simulate_poisson(rg) for _ in range(R)])

# +
# Cox process
print("\nSimulating Cox catastrophe process.\n")

lambda0 = 0.49
a = 0.4
rho = 0.4
delta = 1

selfJumpSizeDist = lambda rg: 0
extJumpSizeDist = lambda rg: rg.uniform(0, 0.5)

def simulate_cox(rg):
    return simulate_num_dynamic_contagion(rg, maturity,
        lambda0, a, rho, delta,
        selfJumpSizeDist, extJumpSizeDist)

rg = np.random.default_rng(123)
np.mean([simulate_cox(rg) for _ in range(R)])

# +
# Hawkes process
print("\nSimulating Hawkes catastrophe process.\n")

lambda0 = 0.47
a = 0.26
rho = 0.4
delta = 1

selfJumpSizeDist = lambda rg: rg.uniform()
extJumpSizeDist = lambda rg: 0

def simulate_hawkes(rg):
    return simulate_num_dynamic_contagion(rg, maturity,
        lambda0, a, rho, delta,
        selfJumpSizeDist, extJumpSizeDist)

rg = np.random.default_rng(123)
np.mean([simulate_hawkes(rg) for _ in range(R)])

# +
# Dynamic contagion process
print("\nSimulating dynamic contagion catastrophe process.\n")

lambda0 = 0.29
a = 0.26
rho = 0.4
delta = 1

selfJumpSizeDist = lambda rg: rg.uniform()
extJumpSizeDist = lambda rg: rg.uniform(0, 0.5)

def simulate_dcp(rg):
    return simulate_num_dynamic_contagion(rg, maturity,
        lambda0, a, rho, delta,
        selfJumpSizeDist, extJumpSizeDist)

rg = np.random.default_rng(123)
np.mean([simulate_dcp(rg) for _ in range(R)])
