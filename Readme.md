# Gaussian Mixture Linear Regression (GMLR) 

Gaussian Mixture Linear Regression (GMLR) model with adjustable state-dependent probability. This package is designed for fitting and predicting mixture models using the Expectation-Maximization (EM) algorithm, specifically tailored for time series data with Markov switching. 

## Features 

- **Markov Switching Gaussian Mixture Model with Lagged Variables (MSGMLVAR)**: Extends the Gaussian Mixture Linear Regression (GMLR) to include support for lagged variables and Markov switching. 

- **MSGMLR**: Fit and predict using a mixture model for time series data with adjustable state-dependent probability.

- **Flexible Data Input**: Accepts any `pandas.DataFrame` as input, with the index of the dataset being the time index.

- **Visualization**: Supports plotting the fitting process and results.

## Regression Models

### GMLR Class

**Description:** The Gaussian Mixture Linear Regression (GMLR) class is designed for fitting and predicting mixture models using the Expectation-Maximization (EM) algorithm.

**Parameters:**

- **data (pd.DataFrame):** Input data.
- **ycol (list):** List of column names for the dependent variable(s).
- **Xcol (list):** List of column names for the independent variable(s).
- **ngroup (int):** Number of groups or states in the model.
- **const (bool):** Boolean indicating whether to include a constant term in the regression model.
- **cov (bool):** Boolean indicating whether to include a covariance term in the variance-covariance matrix.

**Usage Example:**

```python
pythonCopy code# Initialize the GMLR instance
gmlr_model = GMLR(data, ycol=['y1', 'y2'], Xcol=['X1', 'X2'], ngroup=2, const=True)

# Fit the model using the EM algorithm
gmlr_model.fit(maxiter=50, tol=1e-6)

# Generate predictions for the dependent variables
predictions = gmlr_model.predict()
```

### MSGMLR Class

**Description:** The Markov Switching Gaussian Mixture Linear Regression (MSGMLR) class extends the functionality of the GMLR model by providing support for fitting and predicting mixture models for time series data using the Expectation-Maximization (EM) algorithm.

**Parameters:**

- **data (pd.DataFrame):** Input data.
- **ycol (list):** List of column names for the dependent variable(s).
- **Xcol (list):** List of column names for the independent variable(s).
- **ngroup (int):** Number of groups or states in the model.
- **const (bool):** Boolean indicating whether to include a constant term in the regression model.
- **cov (bool):** Boolean indicating whether to include a covariance term in the variance-covariance matrix.

**Usage Example:**

```python
# Initialize the MSGMLR instance
msgmlr_model = MSGMLR(data, ycol=['y1', 'y2'], Xcol=['X1', 'X2'], ngroup=2, const=True)

# Fit the model using the EM algorithm
msgmlr_model.fit(maxiter=50, tol=1e-6)

# Generate predictions for the dependent variables
predictions = msgmlr_model.predict()
```

### MSGMLVAR Class

**Description:** The Markov Switching Gaussian Mixture Model with Lagged Variables (MSGMLVAR) class extends the functionality of the GMLR model to include lagged variables and support for Markov switching.

**Parameters:**

- **data (pd.DataFrame):** Input data.
- **ycol (list):** List of column names for the endogenous variables.
- **Xcol (list):** List of column names for exogenous variables.
- **extXcol (list):** List of column names for externally given exogenous variables.
- **lags (list|int):** Number of lags to consider.
- **ngroup (int):** Number of groups for the Gaussian mixture model.
- **const (bool):** Whether to include a constant term.
- **cov (bool):** Whether to calculate covariances.
- **alpha (float):** Coefficient for regularization.
- **norm (float):** Normalization factor.

**Usage Example:**

```python
# Initialize the MSGMLVAR instance
msgmlvar_model = MSGMLVAR(data, ycol=['y1', 'y2'], Xcol=['X1', 'X2'], ngroup=2, const=True)

# Fit the model using the EM algorithm
msgmlvar_model.fit(maxiter=50, tol=1e-6)

# Generate predictions for the dependent variables
predictions = msgmlvar_model.predict()
```

## Documentation

For more detailed information about each function and class, refer to the comments in the code or the additional documentation provided.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
