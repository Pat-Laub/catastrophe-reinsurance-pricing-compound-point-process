# ---
# jupyter:
#   jupytext:
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
# %config InlineBackend.figure_format='retina'

# Add the parent directory to the path so that we can import the modules.
import os
import sys
from functools import partial

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import pandas as pd

from dynamic_contagion import *
from market_conditions import *
from reinsurance import *

# +
# Other parameters
maturity = 3
markup = 0.4
R = int(1e5)
seed = 123

# Asset parameters
V_0 = 130
phi_V = -3  # * (1.3) # = V0 / L0
sigma_V = 0.05

# Liability parameters
L_0 = 100
phi_L = -3
sigma_L = 0.02

# Interest rate parameters
r_0 = 0.02
k = 0.2
m = 0.05  # 0.5
upsilon = 0.1
eta_r = -0.01

# Catastrophe loss size distribution parameters
mu_C = 2
sigma_C = 0.5


# +
# Poisson process
def simulate_poisson(seed):
    lambda_ = 0.5
    rg = rnd.default_rng(seed)
    return rg.poisson(lambda_ * maturity)


# Cox proces
def simulate_cox(seed):
    lambda0 = 0.49
    a = 0.4
    rho = 0.4
    delta = 1

    return simulate_num_dynamic_contagion_uniform_jumps(
        seed, maturity, lambda0, a, rho, delta, 0.0, 0.0, 0.0, 0.5
    )


# Hawkes process
def simulate_hawkes(seed):
    lambda0 = 0.47
    a = 0.26
    rho = 0.4
    delta = 1

    return simulate_num_dynamic_contagion_uniform_jumps(
        seed, maturity, lambda0, a, rho, delta, 0.0, 1.0, 0.0, 0.0
    )


# Dynamic contagion process
def simulate_dcp(seed):
    lambda0 = 0.29
    a = 0.26
    rho = 0.4
    delta = 1

    return simulate_num_dynamic_contagion_uniform_jumps(
        seed, maturity, lambda0, a, rho, delta, 0.0, 1.0, 0.0, 0.5
    )

simulators = (simulate_poisson, simulate_cox, simulate_hawkes, simulate_dcp)
# -

# ## Tables 1-4

# +
ROUNDING = 4

As = (10.0, 15.0, 20.0, 25.0, 30.0)
Ms = (60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0)

def prices_to_df(prices, As, Ms):
    try:
        cols = [f"$M={int(m)}$" for m in Ms]
        rows = [f"$A={int(a)}$" for a in As]
    except ValueError:
        cols = [f"$M={m}$" for m in Ms]
        rows = [f"$A={a}$" for a in As]

    return pd.DataFrame(prices.round(ROUNDING), columns=cols, index=rows)


def prices_to_tex(prices, As, Ms):
    df = prices_to_df(prices, As, Ms)
    display(df)
    return (
        df.style.to_latex()
            .replace("lrrrrrrr", "c|c|c|c|c|c|c|c")
            .replace("00 ", " ")
    )


# -

# %%time
prices = reinsurance_prices(
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
    simulators,
    mu_C,
    sigma_C,
    markup,
    A=As,
    M=Ms,
)

# Tables 1 to 4
cat_models = ("Poisson", "Cox", "Hawkes", "DCP")
for c in range(4):
    print(cat_models[c])
    print(prices_to_tex(prices[c], As, Ms))

# ## Table 5: Default risk premium

# %%time
risky_prices = reinsurance_prices(
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
    simulators,
    mu_C,
    sigma_C,
    markup,
)[:, np.newaxis]

risky_prices.round(4)

# %%time
safe_prices = reinsurance_prices(
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
    simulators,
    mu_C,
    sigma_C,
    markup,
    defaultable=False,
)[:, np.newaxis]

safe_prices.round(4)

risk_premium = safe_prices - risky_prices
risk_premium.round(4)

np.hstack((risky_prices, safe_prices, risk_premium)).round(4)


# ## Table 6: Impacts of externally-excited jump frequency rate

# +
def simulate_dcp_variations(seed, rho):
    lambda0 = 0.29
    a = 0.26
    delta = 1

    return simulate_num_dynamic_contagion_uniform_jumps(
        seed, maturity, lambda0, a, rho, delta, 0.0, 1.0, 0.0, 0.5
    )


dcp_variations = [partial(simulate_dcp_variations, rho=rho) for rho in (0.4, 3, 10, 20)]
# -

# %%time
risky_prices = reinsurance_prices(
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
    tuple(int(scale * L_0) for scale in (1.1, 1.3, 1.5)),
    L_0,
    r_0,
    dcp_variations,
    mu_C,
    sigma_C,
    markup,
).T

risky_prices.round(4)

# %%time
safe_prices = reinsurance_prices(
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
    tuple(int(scale * L_0) for scale in (1.1, 1.3, 1.5)),
    L_0,
    r_0,
    dcp_variations,
    mu_C,
    sigma_C,
    markup,
    defaultable=False,
).T

safe_prices.round(4)

risk_premium = safe_prices - risky_prices

np.hstack([risky_prices, risk_premium]).round(4)


# ## Table 7: Impacts of externally-excited jump magnitude

# +
def simulate_dcp_variations(seed, mu_F=0.25, mu_G=0.5):
    lambda0 = 0.29
    a = 0.26
    delta = 1
    rho = 0.4

    return simulate_num_dynamic_contagion_uniform_jumps(
        seed, maturity, lambda0, a, rho, delta, 0.0, 2 * mu_G, 0.0, 2 * mu_F
    )


dcp_variations = [
    partial(simulate_dcp_variations, mu_F=mu_F) for mu_F in (0.25, 1.0, 4.0, 8.0)
]

# +
# %%time
risky_prices = reinsurance_prices(
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
    tuple(int(scale * L_0) for scale in (1.1, 1.3, 1.5)),
    L_0,
    r_0,
    dcp_variations,
    mu_C,
    sigma_C,
    markup,
).T

safe_prices = reinsurance_prices(
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
    tuple(int(scale * L_0) for scale in (1.1, 1.3, 1.5)),
    L_0,
    r_0,
    dcp_variations,
    mu_C,
    sigma_C,
    markup,
    defaultable=False,
).T

risk_premium = safe_prices - risky_prices
# -

np.hstack([risky_prices, risk_premium]).round(4)

# ## Table 8: Impacts of self-excited jump magnitude

# +
# %%time

dcp_variations = [
    partial(simulate_dcp_variations, mu_G=mu_G) for mu_G in (0.5, 1.0, 2.0, 3.0)
]

risky_prices = reinsurance_prices(
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
    tuple(int(scale * L_0) for scale in (1.1, 1.3, 1.5)),
    L_0,
    r_0,
    dcp_variations,
    mu_C,
    sigma_C,
    markup,
).T

safe_prices = reinsurance_prices(
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
    tuple(int(scale * L_0) for scale in (1.1, 1.3, 1.5)),
    L_0,
    r_0,
    dcp_variations,
    mu_C,
    sigma_C,
    markup,
    defaultable=False,
).T

risk_premium = safe_prices - risky_prices
# -

np.hstack([risky_prices, risk_premium]).round(4)

# ## Tables 9-12: Prices with catbond

K = 40.0
F = 25.0

# %%time
catbond_prices = reinsurance_prices(
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
    simulators,
    mu_C,
    sigma_C,
    markup,
    A=As,
    M=Ms,
    catbond=True,
    K=K,
    F=F,
)

# +
print("Prices of reinsurance when the reinsurer issues a catbond\n")

for c in range(4):
    print(cat_models[c])
    print(prices_to_tex(catbond_prices[c], As, Ms))
# -

# ## Tables 13-16: Percentage increase in costs after issuing a catbond

# +
print("Percentage increase in price for reinsurance after the reinsurer issues a catbond\n")

for c in range(4):
    print(cat_models[c])
    percentage_increase = ((catbond_prices[c] - prices[c]) / prices[c] * 100).round(2)
    print(prices_to_tex(percentage_increase, As, Ms).replace("00 ", "\% "))
