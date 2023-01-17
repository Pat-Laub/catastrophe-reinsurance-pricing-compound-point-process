# Code accompanying the paper 'Catastrophe reinsurance valuation with CAT bonds using advanced compound point processes'


This repository holds the simulation code written for the submitted paper 'Catastrophe reinsurance valuation with CAT bonds using advanced compound point processes' written by Jiwook Jang and Patrick J. Laub. The paper's tables and figures were generated using this Python code, however an earlier R version of the code is also attached. 



## Python version

Python files:

- `dynamic_contagion.py` has simulation routines for simulating the dynamic contagion process (and hence the other advanced point processes in the paper).
- `market_conditions.py` has methods for simulating the reinsurer's assets, liabilities, and the interest rates according to the CIR-type diffusion process.
- `reinsurance.py` contains the main `reinsurance_prices` method which calculates the value of reinsurance contracts given details of the asset/liability/interest rate processes, a simulator for the catastrophe process, and contract details about the reinsurance contract and the reinsurer's issued catastrophe bonds (if applicable). Many of these inputs accept a list of values, and the method will generate the reinsurance prices over the grid of all combinations of the inputs.



Example usage of the methods are inside the notebooks, e.g. see `notebooks/test_reinsurance.ipynb` for an example of how to use the methods in `reinsurance.py`. In particular, `notebooks/generate_tables.ipynb` contains a notebook which we used to create all the tables in the paper.



## R version

Also attached is an earlier version implemented in R, which can be found in `R/pricing.R`. It can generate reinsurance prices without CAT bonds, and is perhaps easier to read for an R programmer than the final Python implementation.  