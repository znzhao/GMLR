# Gaussian Mixture Linear Regression (GMLR) 

Gaussian Mixture Linear Regression (GMLR) model with adjustable state-dependent probability. This package is designed for fitting and predicting mixture models using the Expectation-Maximization (EM) algorithm, specifically tailored for time series data with Markov switching. 

## Latest Version

- 0.5.0 Sklearn Update (06/18/2024)
  - Add API for global sklearn support
  - Package Restructured for no Matplotlib requirement
  - New graph support for MS-GMLR and MS-GML-VAR models: History Probability Graph

## Features 

- **Markov Switching Gaussian Mixture Model with Lagged Variables (MSGMLVAR)**: Extends the Gaussian Mixture Linear Regression (GMLR) to include support for lagged variables and Markov switching. 

- **MSGMLR**: Fit and predict using a mixture model for time series data with adjustable state-dependent probability.

- **Flexible Data Input**: Accepts any `pandas.DataFrame` as input, with the index of the dataset being the time index.

- **Visualization**: Supports plotting the fitting process and results.

## Regression Models

### GMLR Class

**Description:** The Gaussian Mixture Linear Regression (GMLR) class is designed for fitting and predicting mixture models using the Expectation-Maximization (EM) algorithm.

**Usage Example:**

```python
from gmlr import GMLR
from data.data_generator import PanelGenerator
from distr import CombinedDistr
from sklearn.model_selection import train_test_split

# Sim Data Generation
data = PanelGenerator(ny = 2, ngroup=2, Xrange=(-3,3), seed = 112)
data.summary()
train, test = train_test_split(data.data, test_size = 0.2)

# GMLR Model Fitting
gmlr_mod = GMLR(verbose=3)
X = train[data.Xcol]
y = train[data.ycol]
gmlr_mod.fit(X, y)
gmlr_mod.summary()

# Out-of-Sample Prediction
test = test[data.Xcol]
priors, values, sigmas = gmlr_mod.predict_distr(test)
distr = CombinedDistr(priors, values, sigmas)
distr[0].plot(margin=1, suptitle='Suptitle')
```

### MSGMLR Class

**Description:** The Markov Switching Gaussian Mixture Linear Regression (MSGMLR) class extends the functionality of the GMLR model by providing support for fitting and predicting mixture models for time series data using the Expectation-Maximization (EM) algorithm.

**Usage Example:**

```python
from msgmlr import GMLR
from data.data_generator import TSGenerator
from distr import CombinedDistr

# Sim Data Generation
data = TSGenerator(nX=2, ny = 2, Xrange=(-3, 3), seed=1)
data.summary()
train, test = data.data.iloc[:-5], data.data.iloc[-5:]

# GMLR Model Fitting
msgmlr_mod = MSGMLR(verbose=3)
X = train[data.Xcol]
y = train[data.ycol]
msgmlr_mod.fit(X, y)
msgmlr_mod.summary()

# plot graph for state probability history
msgmlr_mod.plot_history()

# Out-of-Sample Prediction
test = test[data.Xcol]
priors, values, sigmas = msgmlr_mod.predict_distr(test)
distr = CombinedDistr(priors, values, sigmas)
distr[0].plot(margin=1, suptitle='Suptitle')
```

### MSGMLVAR Class

**Description:** The Markov Switching Gaussian Mixture Model with Lagged Variables (MSGMLVAR) class extends the functionality of the GMLR model to include lagged variables and support for Markov switching.

**Usage Example:**
```python
from msgmlvar import MSGMLVAR
from data.data_generator import TSGenerator
from distr import CombinedDistr

# Sim Data Generation
data = TSGenerator(nX=2, ny = 2, Xrange=(-3, 3), seed=1)
data.summary()
train, test = data.data.iloc[:-5], data.data.iloc[-5:]

# GMLR Model Fitting
msgmlvar_mod = MSGMLVAR(verbose=3)
X = train[data.Xcol]
y = train[data.ycol]
msgmlvar_mod.fit(X, y)
msgmlvar_mod.summary()

# plot graph for state impluse response functions
msgmlvar_mod.plot_irf()

# Out-of-Sample Prediction
test = test[data.Xcol]
priors, values, sigmas = msgmlvar_mod.predict_distr(test)
distr = CombinedDistr(priors, values, sigmas)
distr[0].plot(margin=1, suptitle='Suptitle')
```

## Documentation

For more detailed information about each function and class, refer to the comments in the code or the additional documentation provided.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
