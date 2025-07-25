# cqf
Quant notebooks

P directories are the original notebooks utilizing India market instruments

D directories are notebooks re-purposed utilizing TradingStrategy-ai interfaces to DEX instruments

## Installation

Install the client and its dependencies using Poetry:

```
# Assumes you have SSH key set up with your Github account
git clone git@github.com:tradingstrategy-ai/trade-executor.git
cd trade-executor
git submodule update --init --recursive

# Extra dependencies
# - execution: infrastructure to run live strategies
# - web-server: support webhook server of live strategy executors
# - qstrader: still needed to run legacy unit tests
poetry install --all-extras

# Specify Python version
poetry env use python3.11

# Extra miscellaneous dependencies
poetry run pip install pandas cvxpy quantmod utils

# Run commands in the Poetry notebook environment, e.g.:
poetry run ipython ../4.Portfolio\ Optimisation.ipynb

```


