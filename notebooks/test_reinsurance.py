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

# +
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

V_T, L_T, int_r_t = summarise_market_conditions(all_time_series, maturity)
# -

assert np.isnan(all_time_series).mean() == 0


# ## Generate catastrophe scenarios

# +
# Poisson process
def simulate_poisson_slow(seed):
    lambda_ = 0.5
    rg = rnd.default_rng(seed)
    return rg.poisson(lambda_ * maturity)


# Cox proces
def simulate_cox_slow(seed):
    lambda0 = 0.49
    a = 0.4
    rho = 0.4
    delta = 1

    rg = rnd.default_rng(seed)
    selfJumpSizeDist = lambda rg: 0
    extJumpSizeDist = lambda rg: rg.uniform(0, 0.5)

    return simulate_num_dynamic_contagion(
        rg, maturity, lambda0, a, rho, delta, selfJumpSizeDist, extJumpSizeDist
    )


# Hawkes process
def simulate_hawkes_slow(seed):
    lambda0 = 0.47
    a = 0.26
    rho = 0.4
    delta = 1

    rg = rnd.default_rng(seed)
    selfJumpSizeDist = lambda rg: rg.uniform()
    extJumpSizeDist = lambda rg: 0

    return simulate_num_dynamic_contagion(
        rg, maturity, lambda0, a, rho, delta, selfJumpSizeDist, extJumpSizeDist
    )


# Dynamic contagion process
def simulate_dcp_slow(seed):
    lambda0 = 0.29
    a = 0.26
    rho = 0.4
    delta = 1

    rg = rnd.default_rng(seed)
    selfJumpSizeDist = lambda rg: rg.uniform()
    extJumpSizeDist = lambda rg: rg.uniform(0, 0.5)

    return simulate_num_dynamic_contagion(
        rg, maturity, lambda0, a, rho, delta, selfJumpSizeDist, extJumpSizeDist
    )


# +
# %%time
# Catastrophe loss size distribution parameters
mu_C = 2
sigma_C = 0.5

seed = 123

C_T_poisson, num_cats_poisson = simulate_catastrophe_losses(
    seed, R, simulate_poisson_slow, mu_C, sigma_C
)

C_T_cox, num_cats_cox = simulate_catastrophe_losses(
    seed, R, simulate_cox_slow, mu_C, sigma_C
)

C_T_hawkes, num_cats_hawkes = simulate_catastrophe_losses(
    seed, R, simulate_hawkes_slow, mu_C, sigma_C
)

C_T_dcp, num_cats_dcp = simulate_catastrophe_losses(
    seed, R, simulate_dcp_slow, mu_C, sigma_C
)


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


# +
# %%time
# Catastrophe loss size distribution parameters
mu_C = 2
sigma_C = 0.5

seed = 123

C_T_poisson, num_cats_poisson = simulate_catastrophe_losses(
    seed, R, simulate_poisson, mu_C, sigma_C
)

C_T_cox, num_cats_cox = simulate_catastrophe_losses(
    seed, R, simulate_cox, mu_C, sigma_C
)

C_T_hawkes, num_cats_hawkes = simulate_catastrophe_losses(
    seed, R, simulate_hawkes, mu_C, sigma_C
)

C_T_dcp, num_cats_dcp = simulate_catastrophe_losses(
    seed, R, simulate_dcp, mu_C, sigma_C
)
# -

print(
    f"Average number of catastrophes from Poisson process: {np.mean(num_cats_poisson):.2f}"
)
print(f"Average number of catastrophes from Cox process: {np.mean(num_cats_cox):.2f}")
print(
    f"Average number of catastrophes from Hawkes process: {np.mean(num_cats_hawkes):.2f}"
)
print(
    f"Average number of catastrophes from dynamic contagion process: {np.mean(num_cats_dcp):.2f}"
)

print(
    f"Mean/variance catastrophe loss from Poisson process: {np.mean(C_T_poisson):.2f}, {np.var(C_T_poisson):.2f}"
)
print(
    f"Mean/variance catastrophe loss from Cox process: {np.mean(C_T_cox):.2f}, {np.var(C_T_cox):.2f}"
)
print(
    f"Mean/variance catastrophe loss from Hawkes process: {np.mean(C_T_hawkes):.2f}, {np.var(C_T_hawkes):.2f}"
)
print(
    f"Mean/variance catastrophe loss from dynamic contagion process: {np.mean(C_T_dcp):.2f}, {np.var(C_T_dcp):.2f}"
)

print(
    f"Min/max catastrophe loss from Poisson process: {np.min(C_T_poisson):.2f}, {np.max(C_T_poisson):.2f}"
)
print(
    f"Min/max catastrophe loss from Cox process: {np.min(C_T_cox):.2f}, {np.max(C_T_cox):.2f}"
)
print(
    f"Min/max catastrophe loss from Hawkes process: {np.min(C_T_hawkes):.2f}, {np.max(C_T_hawkes):.2f}"
)
print(
    f"Min/max catastrophe loss from dynamic contagion process: {np.min(C_T_dcp):.2f}, {np.max(C_T_dcp):.2f}"
)


# +
# Plot a grouped vertical bar charts showing the number of simulations which had `n` catastrophes according to each of the arrival processes.
def plot_num_cats(num_cats_poisson, num_cats_cox, num_cats_hawkes, num_cats_dcp):
    num_cats_poisson = np.floor(num_cats_poisson)
    num_cats_cox = np.floor(num_cats_cox)
    num_cats_hawkes = np.floor(num_cats_hawkes)
    num_cats_dcp = np.floor(num_cats_dcp)

    max_cats = np.max(
        [
            np.max(num_cats_poisson),
            np.max(num_cats_cox),
            np.max(num_cats_hawkes),
            np.max(num_cats_dcp),
        ]
    )
    min_cats = np.min(
        [
            np.min(num_cats_poisson),
            np.min(num_cats_cox),
            np.min(num_cats_hawkes),
            np.min(num_cats_dcp),
        ]
    )

    num_bins = int(max_cats - min_cats + 1)

    bins = np.linspace(min_cats, max_cats, num_bins + 1)
    bins = np.floor(bins)

    hist_poisson, _ = np.histogram(num_cats_poisson, bins=bins)
    hist_cox, _ = np.histogram(num_cats_cox, bins=bins)
    hist_hawkes, _ = np.histogram(num_cats_hawkes, bins=bins)
    hist_dcp, _ = np.histogram(num_cats_dcp, bins=bins)

    width = 0.2
    x = np.arange(len(bins) - 1)

    plt.bar(x - width, hist_poisson, width, label="Poisson")
    plt.bar(x, hist_cox, width, label="Cox")
    plt.bar(x + width, hist_hawkes, width, label="Hawkes")
    plt.bar(x + 2 * width, hist_dcp, width, label="DCP")

    plt.xticks(x, bins[:-1])
    plt.legend()

    plt.xlim(0.5, 11.0)
    plt.xlabel("Number of catastrophes")
    plt.ylabel("Number of simulations")

    # Change the aspect ratio to be wider
    plt.gcf().set_size_inches(8, 2.5)

    # Remove the top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


plot_num_cats(num_cats_poisson, num_cats_cox, num_cats_hawkes, num_cats_dcp)
plt.savefig("num_catastrophe_hists.png")
# -

# ## Tables 1-4 (LaTeX)

# +
ROUNDING = 4

As = (10.0, 15.0, 20.0, 25.0, 30.0)
Ms = (60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0)


def prices_to_tex(prices, As, Ms):
    try:
        cols = [f"$M={int(m)}$" for m in Ms]
        rows = [f"$A={int(a)}$" for a in As]
    except ValueError:
        cols = [f"$M={m}$" for m in Ms]
        rows = [f"$A={a}$" for a in As]

    df = pd.DataFrame(prices.round(ROUNDING), columns=cols, index=rows)

    display(df)

    return (
        df.style.to_latex().replace("00 ", " ").replace("lrrrrrrr", "c|c|c|c|c|c|c|c")
    )


# -

prices_poisson = calculate_prices(V_T, L_T, int_r_t, C_T_poisson, markup)
print(prices_to_tex(prices_poisson, As, Ms))

prices_cox = calculate_prices(V_T, L_T, int_r_t, C_T_cox, markup)
print(prices_to_tex(prices_cox, As, Ms))

prices_hawkes = calculate_prices(V_T, L_T, int_r_t, C_T_hawkes, markup)
print(prices_to_tex(prices_hawkes, As, Ms))

prices_dcp = calculate_prices(V_T, L_T, int_r_t, C_T_dcp, markup)
print(prices_to_tex(prices_dcp, As, Ms))

price_dcp = calculate_prices(V_T, L_T, int_r_t, C_T_dcp, markup, A=20, M=90)
price_dcp

# +
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
    simulate_dcp,
    mu_C,
    sigma_C,
    markup,
)

prices
