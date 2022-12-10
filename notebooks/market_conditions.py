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
#     display_name: Python [conda env:.conda-catbond]
#     language: python
#     name: conda-env-.conda-catbond-py
# ---

# +
# Add the parent directory to the path so that we can import the modules.
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from main import *
import matplotlib.pyplot as plt

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
kappa = 0.2
m = 0.05  # 0.5
upsilon = 0.1
lambda_r = -0.01

# +
all_time_series = get_market_conditions(R=R, seed=seed, maturity=maturity, kappa=kappa,
        lambda_r=lambda_r, m=m, phi_V=phi_V, sigma_V=sigma_V, phi_L=phi_L, sigma_L=sigma_L,
        upsilon=upsilon, V_0=V_0, L_0=L_0, r_0=r_0)

summarised_time_series = summarise_market_conditions(all_time_series, maturity)

plt.hist(summarised_time_series[:, 0], 30, label="Assets")
plt.axvline(V_0, c="r", ls="--")
plt.legend()
plt.show()

plt.hist(summarised_time_series[:, 1], 30, label="Liabilities")
plt.axvline(L_0, c="r", ls="--")
plt.legend()
plt.show()

plt.hist(summarised_time_series[:, 2], 30, label="Interest rates")
plt.axvline(r_0, c="r", ls="--")
plt.legend()
plt.show()

