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

from market_conditions import *
from dynamic_contagion import *
from reinsurance import *

import matplotlib.pyplot as plt
import pandas as pd

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


# +
# %%time

catbond_fn = lambda C_T, K, F_cat: np.minimum(np.maximum(C_T - K, 0), F_cat)

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
    simulate_poisson,
    mu_C,
    sigma_C,
    markup,
    As = 10.0,
    Ms = 70.0,
	catbond=True,
    K = 40.0,
    psi_T = catbond_fn,
    F_cat = 100.0
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
    simulate_poisson,
    mu_C,
    sigma_C,
    markup,
)


catbond_premium = safe_prices - risky_prices
# -

safe_prices

risky_prices

catbond_premium

delta_0 = catbond_prices(
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
    markup = 0.0,
    As = 10.0,
    Ms = 70.0,
    K = 40.0,
    psi_T = catbond_fn,
    F_cat = 100.0
)


delta_0

# +
# %%time

face_values = np.linspace(0.0, 2, 5)
present_values = np.empty_like(face_values)

for i, face_value in enumerate(face_values):
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
		markup = 0.0,
		As = 10.0,
		Ms = 70.0,
		catbond=True,
		K = 10.0,
		psi_T = catbond_fn,
		F_cat = face_value
	)
# -

present_values

plt.plot(face_values, present_values);
plt.plot(face_values, present_values, ls="--");
# plt.plot(face_values, face_values / face_values[-1] * present_values[-1], ls="--")

# +
free_bond = catbond_prices(
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
        markup = 0.0,
        As = 10.0,
        Ms = 70.0,
        K = 70.0,
        psi_T = catbond_fn,
        F_cat = 0.0
    )
	
free_bond

# +
# %%time
face_values = np.linspace(0.0, 2, 5)
deltas = np.empty_like(face_values)

for i, face_value in enumerate(face_values):
    deltas[i] = catbond_prices(
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
        markup = 0.0,
        As = 10.0,
        Ms = 70.0,
        K = 10.0,
        psi_T = catbond_fn,
        F_cat = face_value
    )
# -

deltas

plt.plot(face_values, deltas);
plt.plot(face_values, face_values, ls="--")
plt.plot(face_values, face_values / face_values[-1] * deltas[-1], ls="--")

# ## Attempt the optimisation problem (Poisson)

# +
catbond_markup = 0.05

def net_present_value(K, F_cat):
	reinsurance_pv = reinsurance_prices(
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
		markup = 0.0,
		As = 10.0,
		Ms = 70.0,
		catbond=True,
		K = K,
		psi_T = catbond_fn,
		F_cat = F_cat
	)

	delta_0 = catbond_prices(
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
		markup = 0.0,
		As = 10.0,
		Ms = 70.0,
		K = K,
		psi_T = catbond_fn,
		F_cat = F_cat
	)

	return markup * reinsurance_pv - catbond_markup * delta_0
# -

net_present_value(10.0, 100.0)

# %%time
A = 10.0
K = A
for F_cat in [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 150.0]:
		print(f'K = {K}, F_cat = {F_cat}, NPV = {net_present_value(K, F_cat)}')


# ## Attempt the optimisation problem (DCP)

# +
catbond_markup = 0.05

def net_present_value(K, F_cat):
	reinsurance_pv = reinsurance_prices(
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
		markup = 0.0,
		As = 10.0,
		Ms = 70.0,
		catbond=True,
		K = K,
		psi_T = catbond_fn,
		F_cat = F_cat
	)

	delta_0 = catbond_prices(
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
		markup = 0.0,
		As = 10.0,
		Ms = 70.0,
		K = K,
		psi_T = catbond_fn,
		F_cat = F_cat
	)

	return markup * reinsurance_pv - catbond_markup * delta_0
# -

net_present_value(10.0, 100.0)

# %%time
A = 10.0
K = A
for F_cat in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:
		print(f'K = {K}, F_cat = {F_cat}, NPV = {net_present_value(K, F_cat)}')


# +
# %%time
no_bond_npv = net_present_value(K, 0.0)

face_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
strikes = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

npvs = np.empty((len(face_values), len(strikes)), dtype = float)

for i, face_value in enumerate(face_values):
	for j, strike in enumerate(strikes):
		npvs[i,j] = net_present_value(strike, face_value)
		if npvs[i,j] > no_bond_npv:
			print(f'K = {K}, F_cat = {F_cat}, NPV = {npvs[i,j]}')
	
# -

npvs.round(4)
