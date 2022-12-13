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

from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from catbond import *
from dynamic_contagion import *
from market_conditions import *
from reinsurance import *

# +
# Other parameters
maturity = 3
markup = 0.4
R = int(1e4)
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

# Catbond (constant) parameters
catbond_markup = 0.05


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


# -

reinsurance_prices_with_catbonds = reinsurance_prices(
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
    (simulate_poisson, simulate_cox, simulate_hawkes, simulate_dcp),
    mu_C,
    sigma_C,
    markup,
    catbond=True,
    K=(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0),
    F=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
)

# +
names = ("Poisson", "Cox", "Hawkes", "DCP")

Fs = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
X = np.array([F / L_0 for F in Fs])
Y = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
X, Y = np.meshgrid(X, Y)

for i, name in enumerate(names):
    print(name)

    Z = reinsurance_prices_with_catbonds[i]

    # Plot a colourful 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    # Plot with the smallest values closest to us
    ax.invert_xaxis()

    # Rotate the plot 40 degrees
    ax.view_init(40, 40)

    plt.xlabel("Face value / L_0")
    plt.ylabel("Strike")

    plt.show()

# +
# Create two 3D subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Plot the first subplot
for i in (0, 3):
    ax = ax1 if i == 0 else ax2
    Z = reinsurance_prices_with_catbonds[i]

    # Plot a colourful 3D surface
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    # Plot with the smallest values closest to us
    ax.invert_xaxis()

    # Rotate the plot 40 degrees
    ax.view_init(40, 40)

    ax.set_xlabel("Face value / L_0")
    ax.set_ylabel("Strike")

    ax.set_title("Poisson" if i == 0 else "DCP")

plt.savefig("poisson-dcp.png", dpi=300, bbox_inches="tight")

# +
# Create two 3D subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Plot the first subplot
for i in (1, 2):
    ax = ax1 if i == 1 else ax2
    Z = reinsurance_prices_with_catbonds[i]

    # Plot a colourful 3D surface
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    # Plot with the smallest values closest to us
    ax.invert_xaxis()

    # Rotate the plot 40 degrees
    ax.view_init(40, 40)

    ax.set_xlabel("Face value / L_0")
    ax.set_ylabel("Strike")

    ax.set_title("Cox" if i == 1 else "Hawkes")

plt.savefig("cox-hawkes.png", dpi=300, bbox_inches="tight")

# +
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
    (simulate_poisson, simulate_cox, simulate_hawkes, simulate_dcp),
    mu_C,
    sigma_C,
    markup,
    catbond=True,
    K=40.0,
    F=10.0,
)

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
    (simulate_poisson, simulate_cox, simulate_hawkes, simulate_dcp),
    mu_C,
    sigma_C,
    markup,
)

catbond_premium = safe_prices - risky_prices
# -

safe_prices

risky_prices

catbond_premium / risky_prices * 100

risky_prices, safe_prices, catbond_premium

# +
# Make a subplot with two square plots horizontally
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot on the left subfigure
for i in range(4):
    ax[0].plot(
        [0, 1], [risky_prices[i], safe_prices[i]], label=names[i], ls="--", marker="*"
    )
ax[0].legend()

# Remove top and right splines
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)

# On ax[0], replace x axis ticks with "Without catastrophe bond", "With catastrophe bond" at 0 and 1
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["No catbond", "With catbond"])

ax[0].set_ylabel("Reinsurance price")

for i in range(4):
    ax[1].plot(
        [0, 1],
        [risky_prices[i] / risky_prices[i], safe_prices[i] / risky_prices[i]],
        label=names[i],
        ls="--",
        marker="*",
    )
ax[1].legend()

# Remove top and right splines
# ax = plt.gca()
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)

# On ax[0], replace x axis ticks with "Without catastrophe bond", "With catastrophe bond" at 0 and 1
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["No catbond", "With catbond"])

ax[1].set_ylabel("Relative prices")

# Add some space between the two subfigures
plt.subplots_adjust(wspace=0.35)

# Increase the font sizes
plt.rcParams.update({"font.size": 14})

plt.savefig("reinsurance_prices_with_and_without_catbonds.png", dpi=300);

# +
for i in range(4):
    plt.plot(
        [0, 1],
        [risky_prices[i] / risky_prices[i], safe_prices[i] / risky_prices[i]],
        label=names[i],
        marker="*",
    )
plt.legend()

# Remove top and right splines
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Replace x axis ticks with "No catbond" and "Catbond" at 0 and 1
plt.xticks([0, 1], ["Without catastrophe bond", "With catastrophe bond"])

plt.ylabel("Normalised price")
# plt.title("Normalised reinsurance prices with & without catbonds");

plt.gcf().set_size_inches(8, 4)

# Increase font size
plt.rcParams.update({"font.size": 14})

# plt.savefig("reinsurance_prices_with_and_without_catbonds.png", dpi=300);
# -

delta_0 = catbond_prices(
    R,
    seed,
    maturity,
    k,
    eta_r,
    m,
    upsilon,
    r_0,
    simulate_poisson,
    mu_C,
    sigma_C,
    markup=0.0,
    K=40.0,
    F=10.0,
)

delta_0

# +
# %%time

Fs = np.linspace(0.0, 2, 5)
present_values = np.empty_like(Fs)

for i, F in enumerate(Fs):
    present_values[i] = reinsurance_prices(
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
        simulate_poisson,
        mu_C,
        sigma_C,
        markup=0.0,
        catbond=True,
        K=20.0,
        F=F,
    )
# -

plt.plot(Fs, present_values)
plt.plot(Fs, present_values, ls="--");

# +
free_bond = catbond_prices(
    R,
    seed,
    maturity,
    k,
    eta_r,
    m,
    upsilon,
    r_0,
    simulate_poisson,
    mu_C,
    sigma_C,
    markup=0.0,
    K=70.0,
    F=0.0,
)

free_bond

# +
Fs = np.linspace(0.0, 2, 5)
deltas = np.empty_like(Fs)

for i, F in enumerate(Fs):
    deltas[i] = catbond_prices(
        R,
        seed,
        maturity,
        k,
        eta_r,
        m,
        upsilon,
        r_0,
        simulate_poisson,
        mu_C,
        sigma_C,
        markup=0.0,
        K=10.0,
        F=F,
    )
# -

deltas

plt.plot(Fs, deltas)
plt.plot(Fs, Fs, ls="--")
plt.plot(Fs, Fs / Fs[-1] * deltas[-1], ls="--");

# ## Attempt the optimisation problem (Poisson)

net_present_value(
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
    simulate_poisson,
    mu_C,
    sigma_C,
    markup,
    catbond_markup,
    K=20.0,
    F=0.0,
)

net_present_value(
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
    simulate_poisson,
    mu_C,
    sigma_C,
    markup,
    catbond_markup,
    K=20.0,
    F=10.0,
)

npv = partial(
    net_present_value,
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
    simulate_poisson,
    mu_C,
    sigma_C,
    markup,
    catbond_markup,
)

# %%time
A = 20.0
K = A
for F in (0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 150.0):
    print(f"K = {K}, F = {F}, NPV = {npv(K, F)}")

# ## Attempt the optimisation problem (DCP)

# +
npv_dcp = partial(
    net_present_value,
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
    catbond_markup,
)

npv_dcp(K=20.0, F=10.0)
# -

# %%time
A = 20.0
K = A
for F in (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0):
    print(f"K = {K}, F = {F}, NPV = {npv_dcp(K, F)}")

# +
# %%time
no_bond_npv = npv_dcp(K=40.0, F=0.0)

Ks = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)
Fs = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)

npvs = np.empty((len(Ks), len(Fs)), dtype=float)

for i, K in enumerate(tqdm(Ks)):
    for j, F in enumerate(Fs):
        npvs[i, j] = npv_dcp(K, F)
# -

npvs.round(4)

df = pd.DataFrame(npvs, columns=Fs, index=Ks)
df.index.name = "K"
df

df == df.max().max()

diff = df - no_bond_npv
diff

binary_diff = diff.copy()
binary_diff[binary_diff > 0] = 1
binary_diff[binary_diff < 0] = 0
binary_diff

# +
X = Ks
Y = Fs
X, Y = np.meshgrid(X, Y)
Z = df.to_numpy().T

# Plot the surface.
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.contour3D(X, Y, Z, 50, cmap="binary")

plt.ylabel("Face value")
plt.xlabel("Strike");
# -

K_reins_opt = 50.0
F_reins_opt = 40.0

# ## Insurer's point of view

# +
thc = partial(
    total_hedging_cost,
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
    catbond_markup,
    K_reins=K_reins_opt,
    F_reins=F_reins_opt,
)

thc(K_ins=20.0, F_ins=10.0)

# +
# %%time
no_bond_thc = thc(A, 0.0)

Ks = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)
Fs = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)

thcs = np.empty((len(Ks), len(Fs)), dtype=float)

for i, K in enumerate(tqdm(Ks)):
    for j, F in enumerate(Fs):
        thcs[i, j] = thc(K, F)
