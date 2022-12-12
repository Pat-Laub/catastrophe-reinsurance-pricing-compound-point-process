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

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from market_conditions import *
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

plt.hist(V_T, 30, label="Assets")
plt.axvline(V_0, c="r", ls="--")
plt.legend()
plt.show()

plt.hist(L_T, 30, label="Liabilities")
plt.axvline(L_0, c="r", ls="--")
plt.legend()
plt.show()

plt.hist(int_r_t, 30, label="Interest rates (integrated)")
plt.axvline(r_0, c="r", ls="--")
plt.legend()
plt.show()
