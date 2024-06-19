import copy
import numpy as np
import pandas as pd
from tqdm import trange
from typing import Literal
from api.sklearn_msgmlr import SklearnMSGMLR
from sklearn.utils.validation import check_is_fitted
from helper.utils import gradient

class SklearnMSGMLVAR(SklearnMSGMLR):
    _parameter_constraints = {
        "x_lags": [int, list],
        "y_lags": [int, list],
        "include_curr_x": [bool],
        "const_col_index": [list, str, None],
        "n_clusters": [int],
        "fit_intercept": [bool],
        "fit_cov": [bool],
        "alpha": [float],
        "norm": [int, float],
        "warm_start": [bool],
        "path": [str, None],
        "max_iter": [int],
        "tol": [float],
        "step_plot":[callable, None],
        "verbose": [int],
        "pred_mode": [str]
    }
    def __init__(self, x_lags:int|list[int] = 1, y_lags:int|list[int] = 1, include_curr_x: bool = True, const_col_index: list[int]|Literal['all'] = None, 
                 ascending: bool = True, n_clusters: int = 2, fit_intercept: bool = True, fit_cov: bool = False, alpha: float = 0.0, 
                 norm: int|float = 1, warm_start: bool = False, path: str = None, max_iter:int = 100,
                 tol:float = 1e-4, step_plot: callable = None, pred_mode: Literal['naive', 'loglik'] = 'naive', verbose = 0):
        """
        A Markov Switching Gaussian Mixture Linear Vector Auto Regression (MS-GML-VAR) model compatible with scikit-learn.

        This model fits a Markov Switching Gaussian mixture model to the vector data, allowing for 
        a flexible representation of the underlying distributions.

        Parameters
        ----------
        x_lags : int, default=1
            Number of lagged values for X.
        y_lags : int, default=1
            Number of lagged values for y.
        include_curr_x : bool, default=True
            Flag to include current X values in modeling.
        const_col_index : list of int or Literal['all'], optional
            Indices of columns in X to treat as constants or 'all' columns.
        ascending : bool, default=True
            Whether to process data in ascending time order.
        n_clusters : int, default=2
            Number of clusters for modeling.
        fit_intercept : bool, default=True
            Whether to fit an intercept term.
        fit_cov : bool, default=False
            Whether to fit covariance terms.
        alpha : float, default=0.0
            Regularization parameter.
        norm : int or float, default=1
            panelty regularization parameter. Default is L1 panelty.
        warm_start : bool, default=False
            Whether to reuse the solution of the previous call to fit.
        path : str or None, default=None
            Path to save or read model configurations.
        max_iter : int, default=100
            Maximum number of iterations.
        tol : float, default=1e-4
            Tolerance for convergence.
        step_plot : callable or None, optional
            Callable function for plotting steps during fitting.
        pred_mode : Literal['naive', 'loglik'], default='naive'
            Prediction mode: 'naive' or 'loglik'.
        verbose : int, default=0
            Verbosity level.

        Attributes
        ----------
        is_fitted_ : bool
            A boolean indicating whether the estimator has been fitted.

        feature_ : ndarray
            Names of features seen during :term:`fit`.

        target_ : ndarray
            Names of targets seen during :term:`fit`.

        Examples
        --------
        >>> from gmlr.api.msgmlvar import SklearnMSGMLVAR
        >>> import numpy as np
        >>> X = np.arange(100).reshape(100, 1)
        >>> y = np.zeros((100, ))
        >>> msgmlvar_mod = SklearnMSGMLVAR()
        >>> msgmlvar_mod.fit(X, y)
        """
        if type(x_lags) == int and x_lags < 1:
            raise ValueError("Cannot use future value to predict current values")
        if type(y_lags) == int and y_lags < 0:
            raise ValueError("Cannot use future value to predict current values")
        self.x_lags = x_lags
        self.y_lags = y_lags
        self.include_curr_x = include_curr_x
        self.const_col_index = const_col_index
        super().__init__(ascending, n_clusters, fit_intercept, fit_cov, alpha, norm, warm_start, path, max_iter, tol, step_plot, pred_mode, verbose)
    
    def saveConfig(self, thetas, path: str = './config/msgmlvar_config.npy'):
        """
        Save model configurations to a file.

        Parameters
        ----------
        thetas : array-like
            Model parameters to be saved.
        path : str, default='./config/msgmlvar_config.npy'
            File path to save configurations.
        """
        super.saveConfig(thetas, path)

    def readConfig(self, path: str = './config/msgmlvar_config.npy'):
        """
        Read model configurations from a file.

        Parameters
        ----------
        path : str, default='./config/msgmlvar_config.npy'
            File path to read configurations from.

        Returns
        -------
        array-like
            Read model parameters.
        """
        return super.readConfig(path)

    def process(self, X, y):
        """
        Process input data for model fitting.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            Target values.

        Returns
        -------
        array-like, array-like, list, list
            Processed X data, processed y data, columns to lag for X, columns not to lag for X.
        """
        X_to_model = None
        # include the lags for y
        if not self.ascending:
            y = y[::-1]
        X_to_model = None
        # generate lags
        if self.y_lags >= 1:
            for lag in range(1, self.y_lags+1):
                lag_y = np.zeros(y.shape)*np.nan
                lag_y[lag:] = copy.deepcopy(y[:-lag])
                X_to_model = copy.deepcopy(lag_y) if X_to_model is None else np.hstack([X_to_model, lag_y])
        if not self.ascending:
            y = y[::-1]
            X_to_model = X_to_model[::-1]

        # include the lags for X
        if X is not None:
            if self.const_col_index is None:
                to_lag_col = list(range(X.shape[1]))
                no_lag_col = []
            elif self.const_col_index == 'all':
                to_lag_col = []
                no_lag_col = list(range(X.shape[1]))
            else:
                to_lag_col = list(set(range(X.shape[1])) - set(self.const_col_index))
                no_lag_col = self.const_col_index

            if not self.ascending:
                X = X[::-1]
                X_to_model = X_to_model[::-1]

            to_lag = X[:, to_lag_col]
            no_lag = X[:, no_lag_col]
            # X to regression
            if self.include_curr_x:
                X_to_model = copy.deepcopy(to_lag) if X_to_model is None else np.hstack([X_to_model, to_lag])
                
            # generate lags
            if self.x_lags >=1:
                for lag in range(1, self.x_lags+1):
                    lag_X = np.zeros(to_lag.shape)*np.nan
                    lag_X[lag:] = copy.deepcopy(to_lag[:-lag])
                    X_to_model = copy.deepcopy(lag_X) if X_to_model is None else np.hstack([X_to_model, lag_X])

            X_to_model = copy.deepcopy(no_lag) if X_to_model is None else np.hstack([X_to_model, no_lag])
            if not self.ascending:
                X = X[::-1]
                X_to_model = X_to_model[::-1]

        return X_to_model, y, to_lag_col, no_lag_col
    
    def fit(self, X = None, y = None):
        """
        Fit the MSGMLVAR model to the provided data.

        Parameters
        ----------
        X : array-like, default=None
            Input features.
        y : array-like, default=None
            Target values.

        Returns
        -------
        self
            Fitted model instance.
        """
        if y is None:
            raise ValueError('X and y cannot be both empty.')
        
        if type(y) == pd.DataFrame:
            target_ = list(y.columns)
        
        if X is None:
            y = self._validate_data(y, accept_sparse=False, multi_output=True)
        else:
            if type(X) == pd.DataFrame:
                features_ = list(X.columns) + ['Const'] if self.fit_intercept else list(X.columns)
            X, y = self._validate_data(X, y, accept_sparse=False, multi_output=True)
        
        X = X.astype(float)
        y = y.astype(float)

        if not hasattr(self, 'features_'):
            features_ = ['X'+str(i) for i in range(X.shape[1])]
        if not hasattr(self, 'target_'):
            if y.ndim == 1:
                target_ = ['y0']
            else:
                target_ = ['y'+str(i) for i in range(y.shape[1])]


        X_to_model, y, to_lag_col, no_lag_col = self.process(X, y)
        self.input_X_ = X
        self.input_y_ = y
        
        X_to_model_col = []
        # generate lags
        y_lags = self.y_lags if type(self.y_lags) == list else range(1, self.y_lags+1)
        if (type(self.y_lags) == int and self.y_lags >= 1) or type(self.y_lags) == list:
            for lag in y_lags:
                X_to_model_col += ['L{}.{}'.format(lag, y_label) for y_label in target_]
        # include the lags for X
        if X is not None:
            if self.include_curr_x:
                X_to_model_col += [features_[k] for k in to_lag_col]
            # generate lags
            if self.x_lags >=1:
                for lag in range(1, self.x_lags+1):
                    X_to_model_col += ['L{}.{}'.format(lag, features_[k]) for k in to_lag_col]
            X_to_model_col += [features_[k] for k in no_lag_col]

        # put them into dataframe
        X_to_model = pd.DataFrame(X_to_model, columns=X_to_model_col)
        y = pd.DataFrame(y, columns=target_)
        data = pd.concat([X_to_model, y], axis=1).dropna(axis=0)
        self.X_to_model_col_ = X_to_model_col    
        super().fit(data[X_to_model_col], data[target_])
        return self

    def predict(self, X = None, pred_std: bool = False):
        """
        Predict future values using the fitted model.

        Args:
            X : array-like or None, default=None
                Input features. If None, predict using internal input_X_.
            pred_std : bool, default=False
                Whether to return prediction standard deviations.

        Returns:
            np.array or tuple
                Predicted values or tuple of predicted values and standard deviations if pred_std=True.
        """
        check_is_fitted(self)
        if X is None:
            return super().predict(X, pred_std)
        else:
            if type(X) == pd.DataFrame:
                X = X.to_numpy()
            if type(X) == list:
                X = np.array(X)
            n_pred = X.shape[0]
            input_X_ = self.input_X_
            input_y_ = self.input_y_

            verbose, self.verbose = self.verbose, 0
            if verbose > 0: print('Model Prediction Started...')
            stds = []
            for t in trange(n_pred, disable = verbose < 2):
                input_X_ = np.vstack([input_X_, X[t] ])
                input_y_ = np.vstack([input_y_, np.nan*np.zeros((1, self.n_y_))])
                X_to_model, y, to_lag_col, no_lag_col = self.process(input_X_, input_y_)
                X_to_model = X_to_model[-1][np.newaxis, :]
                if pred_std:
                    input_y_[-1], std = super().predict(X_to_model, pred_std)
                    stds.append(std[0])
                else:
                    input_y_[-1] = super().predict(X_to_model, pred_std)
                    
            self.verbose = verbose
            if pred_std:
                return input_y_[-n_pred:], np.array(stds)
            else:
                return input_y_[-n_pred:]
                

    def irf(self, state = 0, periods = 5, thetas = None):
        """
        Predict state IRF for a given number of periods.

        Args:
            state (int): State to predict.
            periods (int): Number of periods to predict.

        Returns:
            irf (np.array): Predicted state IRF for each period.
        """
        if thetas is None:
            thetas = self.thetas_
        gammas, etas, betas, sigmas = self.unpack(thetas)
        irf = np.zeros((periods, self.n_y_, self.n_y_))
        irf[0, :, :] = np.eye(self.n_y_)
        beta = betas[state, :, :]
        beta_lags = np.zeros((self.y_lags, self.n_y_, self.n_y_))
        for lag in range(self.y_lags):
            beta_lags[lag, :, :] = beta[self.n_y_*(lag):self.n_y_*(lag+1), :]
        for p in range(1, periods):
            for lag in range(min(self.y_lags, p)):
                irf[p, :, :] += irf[p-lag-1, :, :].dot(beta_lags[lag, :, :])
        return irf
    
    def irf_std(self, state = 0, periods = 5):
        """
        Calculate standard deviation of state prediction.

        Args:
            state (int): State to predict.
            periods (int): Number of periods.

        Returns:
            stds (np.array): Standard deviations of state prediction for each period.
        """
        grad_y = gradient(self.thetas_, lambda thetas: self.irf(state, periods, thetas))
        stds = np.zeros((periods, self.n_y_, self.n_y_))
        for p in range(periods):
            predvars = np.zeros((self.n_y_, self.n_y_))
            for shock_i in range(self.n_y_):
                for y_i in range(self.n_y_):
                    predvars[shock_i, y_i] = grad_y[:, p, shock_i, y_i].T.dot(self.var_thetas_).dot(grad_y[:, p, shock_i, y_i])
            stds[p, :, :] = np.sqrt(predvars)
        return stds
