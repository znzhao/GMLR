"""
Gaussian Mixed Linear Regression for Sklearn toolkit API
"""

# Author: Zhenning Zhao
# Department of Economics
# University of Texas at Austin
# Last Update: 06/13/2024

import os
import time
import numpy as np
import pandas as pd
import tqdm
from typing import Literal
import scipy.stats as stats
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils.validation import check_is_fitted
from helper.utils import Timer, gradient


import warnings
warnings.filterwarnings('ignore')

class SklearnGMLR(BaseEstimator):
    """
    A Gaussian Mixture Linear Regression (GMLR) model compatible with scikit-learn.

    This model fits a Gaussian mixture model to the data, allowing for 
    a flexible representation of the underlying distributions.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    fit_cov : bool, default=False
        Whether to fit the covariance matrix.

    alpha : float, default=0.0
        Regularization strength; must be a positive float.

    norm : int or float, default=1
        The norm to use for regularization.

    warm_start : bool, default=False
        Whether to use the solution of the previous call to fit as initialization.

    path : str or None, default=None
        The path to save the model configuration.

    max_iter : int, default=100
        Maximum number of iterations of the optimization algorithm.

    tol : float, default=1e-4
        Tolerance for the optimization.

    step_plot : callable or None, default=None
        Function to plot the progress of the algorithm.

    verbose : int, default=0
        The verbosity level.

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
    >>> from gmlr.api.gmlr import SklearnGMLR
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> gmlr_mod = SklearnGMLR()
    >>> gmlr_mod.fit(X, y)
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
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

    def __init__(self, n_clusters: int = 2, fit_intercept: bool = True, fit_cov: bool = False, alpha: float = 0.0, 
                 norm: int|float = 1, warm_start: bool = False, path: str = None, max_iter:int = 100,
                 tol:float = 1e-4, step_plot: callable = None, pred_mode: Literal['naive', 'loglik'] = 'naive', verbose = 0):
        self.n_clusters = n_clusters
        self.fit_intercept = fit_intercept
        self.fit_cov = fit_cov
        self.alpha = alpha
        self.norm = norm
        self.warm_start = warm_start
        self.path = path
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.step_plot = step_plot
        self.pred_mode = pred_mode

    def model_init(self):
        """
        Initializes the model parameters and structure.
        """
        self.n_obs_ = self.X_.shape[0]
        
        self.X_ = np.hstack((self.X_, np.ones(shape=(self.n_obs_, 1)))) if self.fit_intercept else self.X_
        if self.y_.ndim == 1:
            self.y_ = self.y_.reshape(self.y_.shape[0], 1)
        elif self.y_.ndim > 2:
            raise ValueError("Target values must be 1D or 2D.")

        self.n_X_ = self.X_.shape[1]
        self.n_y_ = self.y_.shape[1]
        self.n_gammas_ = (self.n_clusters-1)*self.n_X_
        self.n_betas_ = self.n_X_*self.n_clusters*self.n_y_
        self.n_sigmas_ = self.n_clusters*self.n_y_
        self.n_covs_ = self.n_clusters*int(self.n_y_*(self.n_y_-1)/2)
        self.slices_ = dict()
        self.slices_['gammas'] = np.index_exp[:self.n_gammas_]
        self.slices_['betas'] = np.index_exp[self.n_gammas_: self.n_gammas_ + self.n_betas_]
        self.slices_['sigmas'] = np.index_exp[self.n_gammas_ + self.n_betas_: self.n_gammas_ + self.n_betas_ + self.n_sigmas_]
        self.slices_['covs'] = np.index_exp[self.n_gammas_ + self.n_betas_ + self.n_sigmas_: self.n_gammas_ + self.n_betas_ + self.n_sigmas_ + self.n_covs_]
    
    def save_config(self, path: str = './config/gmlr_config.npy'):
        """
        Saves the current model configuration to a file.

        Parameters
        ----------
        path : str, default='./config/gmlr_config.npy'
            Path to the file where the model configuration will be saved.
        """
        check_is_fitted(self)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as configfile:
            np.save(configfile, self.thetas_)

    def read_config(self, path: str = './config/gmlr_config.npy'):
        """
        Reads the model configuration from a file.

        Parameters
        ----------
        path : str, default='./config/gmlr_config.npy'
            Path to the file from which the model configuration will be read.

        Returns
        -------
        thetas : np.array
            Array of model parameters.
        """
        thetas = np.load(path)
        return thetas

    def unpack(self, thetas: np.array):
        """
        Unpacks the flattened parameter array into individual matrices for further use.

        Parameters
        ----------
        thetas : np.array
            Numpy array containing the parameters of the model.

        Returns
        -------
        gammas : np.array
            Unpacked gamma parameters.

        betas : np.array
            Unpacked beta parameters.

        sigmas : np.array
            Unpacked sigma parameters.
        """
        if self.fit_cov:
            assert len(thetas) == self.n_gammas_ + self.n_betas_ + self.n_sigmas_ + self.n_covs_
        else:
            assert len(thetas) == self.n_gammas_ + self.n_betas_ + self.n_sigmas_

        gammas = thetas[self.slices_['gammas']]
        gammas = np.reshape(gammas, newshape=(self.n_clusters-1, self.n_X_))
        betas = thetas[self.slices_['betas']]
        betas = np.reshape(betas, newshape=(self.n_clusters, self.n_X_, self.n_y_))
        sigmas = thetas[self.slices_['sigmas']]
        sigmas = np.reshape(sigmas, newshape=(self.n_clusters, self.n_y_))
        sigmas = np.array([np.diag(sigmas[g]) for g in range(self.n_clusters)])

        if self.fit_cov and int(self.n_y_*(self.n_y_-1)/2)>0:
            # lower trianglular decomposition factors
            covs = thetas[self.slices_['covs']]
            covs = np.reshape(covs, newshape=(self.n_clusters, int(self.n_y_*(self.n_y_-1)/2)))
            for g in range(self.n_clusters):
                k = 0
                for i in range(1, self.n_y_):
                    for j in range(0, i):
                        sigmas[g, i, j] = covs[g, k]
                        k += 1
        for g in range(self.n_clusters):
            sigmas[g] = sigmas[g].dot(sigmas[g].T)
        return gammas, betas, sigmas
    
    def prior(self, X: np.array, thetas: np.array):
        """
        Computes the prior probabilities used in both E-step and M-step.

        Parameters:
        - X: Numpy array of independent variables.
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - Matrix of prior probabilities for each observation belonging to each group.
        """
        gammas, betas, sigmas = self.unpack(thetas)
        # initialize pmtx
        logit_denominator = 1 + np.sum(np.exp(X.dot(gammas.T)), axis = 1)
        logit_denominator = np.repeat(logit_denominator[:, np.newaxis], repeats = self.n_clusters, axis = 1)
        logit_nominator = np.hstack((np.exp(X.dot(gammas.T)), np.ones((X.shape[0], 1))))
        logit = logit_nominator / logit_denominator
        return logit
    
    def partial_lik(self, X: np.array, y: np.array, thetas: np.array):
        """
        Computes the partial likelihood function for the model.

        Parameters
        ----------
        X : np.array
            Numpy array of independent variables.

        y : np.array
            Numpy array of dependent variables.

        thetas : np.array
            Numpy array containing the parameters of the model.

        Returns
        -------
        loglik : np.array
            Matrix of partial likelihood values.
        """
        gammas, betas, sigmas = self.unpack(thetas)
        logit = self.prior(X, thetas)
        y = y.reshape(X.shape[0], self.n_y_)
        ymtx = np.repeat(y[:, :, np.newaxis], self.n_clusters, axis=2)
        err = ymtx - X.dot(betas.T)
        
        pmtx = []
        for g in range(self.n_clusters):
            # Probability that observation falls in the i-th group
            try:
                normal_prob = np.exp(-1.0/2.0 * np.diag(err[:,:,g].dot(np.linalg.inv(sigmas[g])).dot(err[:,:,g].T)))
            except:
                Warning("Singular matrix. Trying Penrose inverse.")
                normal_prob = np.exp(-1.0/2.0 * np.diag(err[:,:,g].dot(np.linalg.pinv(sigmas[g])).dot(err[:,:,g].T)))
                
            pval = logit[:,g] / np.sqrt(2.0*np.pi*np.linalg.det(sigmas[g])) * normal_prob
            pmtx.append(pval)
        loglik = np.array(pmtx).T
        return loglik
    
    def postior(self, X: np.array, y: np.array, thetas: np.array):
        """
        Computes the posterior probabilities for the model.

        Parameters
        ----------
        X : np.array
            Numpy array of independent variables.

        y : np.array
            Numpy array of dependent variables.

        thetas : np.array
            Numpy array containing the parameters of the model.

        Returns
        -------
        posterior_probs : np.array
            Matrix of posterior probabilities.
        """
        # expectation step: for each observation, calculate the probability to fall in group G
        prob = self.partial_lik(X, y, thetas)
        # update probability according to Bayesian Rule
        return prob / np.repeat(prob.sum(axis=1)[:, np.newaxis], self.n_clusters, axis=1)
    
    def log_lik(self, X: np.array, y: np.array, thetas: np.array, probs: np.array):
        """
        Computes the log-likelihood function.

        Input:
        - X: Numpy array of independent variables.
        - y: Numpy array of dependent variables.
        - thetas: Numpy array containing the parameters of the model.
        - probs: Numpy array containing probabilities.

        Output:
        - Log-likelihood value.
        """
        # take expectation of the log-likilihood function
        log_lik = np.sum(probs * np.log(self.partial_lik(X, y, thetas)))
        return log_lik
    
    def penalty(self, thetas: np.array):
        # Calculate the penalty term for the regularization
        # using the specified norm
        return np.nansum(abs(thetas)**self.norm)**(1/self.norm)

    def guess(self):
        # Generate an initial guess for the model parameters
        # initialize the first guess as the OLS regression beta using only a slice of the data
        def gen_guess():
            if self.fit_cov:
                guess = np.zeros(self.n_gammas_ + self.n_betas_ + self.n_sigmas_ + self.n_covs_)
            else:
                guess = np.zeros(self.n_gammas_ + self.n_betas_ + self.n_sigmas_)

            betas = []
            covs = []
            for g in range(self.n_clusters):
                y = self.y_[g*int(self.n_obs_/self.n_clusters): (g+1)*int(self.n_obs_/self.n_clusters)]            
                X = self.X_[g*int(self.n_obs_/self.n_clusters): (g+1)*int(self.n_obs_/self.n_clusters)]
                cov = np.cov(y.T)
                beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
                betas.append(beta)
                covs.append(cov)
            betas = np.stack(betas,axis=0)
            guess[self.slices_['betas']] = np.reshape(betas, self.n_betas_)
            if self.n_y_ > 1:
                guess[self.slices_['sigmas']] = np.stack([np.diag(cov) for cov in covs], axis=0).reshape(self.n_sigmas_)
            else:
                guess[self.slices_['sigmas']] = np.stack(covs, axis=0).reshape(self.n_sigmas_)
            return guess
        
        if (not self.warm_start) or (self.path is None) or (not os.path.exists(self.path)):
            guess = gen_guess()
        else:
            try:
                guess = self.read_config(self.path)
                if self.fit_cov:
                    assert len(guess) == self.n_gammas_ + self.n_betas_ + self.n_sigmas_ + self.n_covs_
                else:
                    assert len(guess) == self.n_gammas_ + self.n_betas_ + self.n_sigmas_

            except:
                guess = gen_guess()
        return guess

    def model_fit(self, plot: callable = None):
        """
        Fits the model using the EM algorithm.

        Returns:
        - Numpy array of final estimated parameters.
        """
        guess = self.guess()
        # lower bound
        lb = np.array([-np.inf]*len(guess))
        lb[self.slices_['sigmas']] = 1e-10
        bnds = Bounds(lb)

        if self.verbose > 0:
            print('EM Estimation Started...') 
        
        thetas = guess
        start = time.time()
        gap = np.inf
        for stepi in range(self.max_iter):
            postiors = self.postior(self.X_, self.y_, thetas)
            res = minimize(lambda thetas: -self.log_lik(self.X_, self.y_, thetas, postiors) + self.alpha * self.penalty(thetas), 
                           thetas, method = 'SLSQP', bounds=bnds, options={'disp': False})
            gap = np.sum(np.abs(res.x - thetas))
            thetas = res.x
            if self.verbose > 1:
                print('Step {}: Log-likeihood = {:.4f}, gap = {:.4f}'.format(stepi, -res.fun, gap))
                if self.verbose > 2 and plot is not None:
                    plot(thetas)
            flag = res.success
            if gap < self.tol:
                break
            if stepi == self.max_iter-1:
                Warning('Maximum number of iteration reached.')
                flag = False
        log_lik_val = self.log_lik(self.X_, self.y_, thetas, postiors)
        end = time.time()
        if self.verbose > 0:
            print('EM Estimation Completed in {:.4f} seconds.'.format(end-start)) 
        return thetas, log_lik_val, flag

    def hessian(self, X: np.array, y: np.array, thetas: np.array, probs: np.array):
        """
        Compute the Hessian matrix.

        Args:
        - X: Input features.
        - y: Target labels.
        - thetas: Array of model parameters.
        - probs: Posterior probabilities.

        Returns:
        - Computed Hessian matrix.
        """
        log_lik_gradient = gradient(thetas, lambda thetas: np.sum(probs * np.log(self.partial_lik(X, y, thetas)), axis=1) )
        hex = np.zeros((len(thetas), len(thetas)))
        for i in range(self.n_obs_):
            hex += log_lik_gradient[:,i][:, np.newaxis].dot(log_lik_gradient[:,i][:, np.newaxis].T)
        return hex/self.n_obs_

    def bic(self, log_lik: float, thetas: np.array):
        """
        Computes the Bayesian Information Criterion (BIC) for the model.

        Input:
        - thetas: Numpy array containing the parameters of the model.

        Output:
        - BIC value.
        """
        k_theta = len(thetas)
        log_n = np.log(self.n_obs_)
        bic_val = k_theta*log_n - 2*log_lik
        return bic_val

    def delta_method(self, thetas: np.array, var_thetas: np.array):
        """
        Computes the variance covariance matrix using delta method.

        Input:
        - thetas: Numpy array containing the parameters of the model.
        - varthetas: 2-D Numpy array containing the variance covariance matrix of the model parameters.

        Output:
        - variance covariance matrix,
        """
        delta = gradient(thetas, lambda thetas: self.unpack(thetas)[-1])
        if self.fit_cov:
            delta = np.concatenate([delta[self.slices_['sigmas']], delta[self.slices_['covs']]], axis = 0)  
        else:
            delta = delta[self.slices_['sigmas']]
        
        var_cov_vecs = []
        for g in range(self.n_clusters):
            delta_g = delta[:, g, :, :].reshape(delta.shape[0], self.n_y_*self.n_y_)
            var_mtx = var_thetas[-delta.shape[0]:, -delta.shape[0]:]
            var_cov_vec = np.diag(delta_g.T.dot(var_mtx).dot(delta_g)).reshape(self.n_y_, self.n_y_)
            var_cov_vecs.append(var_cov_vec)
        return np.array(var_cov_vecs)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fits the model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        if type(X) == pd.DataFrame:
            self.features_ = list(X.columns) + ['Const'] if self.fit_intercept else list(X.columns)
        
        if type(y) == pd.DataFrame:
            self.target_ = list(y.columns)
        
        X, y = self._validate_data(X, y, accept_sparse=False, multi_output=True)
        self.X_ = X.astype(float)
        self.y_ = y.astype(float)

        if not hasattr(self, 'features_'):
            self.features_ = ['X'+str(i) for i in range(X.shape[1])]
            self.features_ = self.features_ + ['Const'] if self.fit_intercept else self.features_
        
        self.model_init()
        
        if not hasattr(self, 'target_'):
            if y.ndim == 1:
                self.target_ = ['y0']
            else:
                self.target_ = ['y'+str(i) for i in range(y.shape[1])]

        if self.n_obs_ <= 5:
            raise ValueError("Too few data points. Current n_samples={}".format(self.n_obs_))
        
        self.thetas_, self.log_lik_val_, self.flag_ = self.model_fit(plot = self.step_plot)
        
        self.is_fitted_ = True
        self.priors_ = self.prior(self.X_, self.thetas_)
        self.postiors_ = self.postior(self.X_, self.y_, self.thetas_)
        hex = self.hessian(self.X_, self.y_, self.thetas_, self.postiors_)
        self.var_thetas_ = np.linalg.inv(hex)/(self.n_obs_ - len(self.thetas_))
        self.bic_val_ = self.bic(self.log_lik_val_, self.thetas_)
        # `fit` should always return `self`
        return self

    def predict(self, X, pred_std: bool = False):
        """
        Makes predictions using the fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        pred_std : bool, default=False
            If True, return standard deviation of the predictions.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Predicted values.
        """
        # Check if fit had been called
        check_is_fitted(self)
        if X is None:
            X = self.X_
        else:
            X = self._validate_data(X, accept_sparse=False, reset=False)
            X = np.hstack((X, np.ones(shape=(X.shape[0], 1)))) if self.fit_intercept else X
        y = []
        if pred_std: 
            stds = []
        if self.verbose > 0:
            print('Model Prediction Started...')
        with Timer("Prediction", display=self.verbose > 0):
            for x_i in tqdm.trange(X.shape[0], disable = self.verbose == 0, desc = 'Model Prediction', leave = False):
                if self.pred_mode == 'naive':
                    pred_func = lambda x_i, X, thetas: self.__pred_func_naive(x_i, X, thetas) 
                else:
                    pred_func = lambda x_i, X, thetas: self.__pred_func(x_i, X, thetas) 
                y_i = pred_func(x_i, X, self.thetas_)
                y.append(y_i)
                if pred_std: 
                    grad_y_i = gradient(self.thetas_, lambda thetas: pred_func(x_i, X, thetas))
                    pred_vars = grad_y_i.T.dot(self.var_thetas_).dot(grad_y_i)
                    stds.append(np.sqrt(np.diag(pred_vars)))
        preds = np.array(y)
        if pred_std:
            stds = np.array(stds)
            return preds, stds
        else:
            return preds
    
    def __pred_func(self, x_i: int, X: np.array, thetas: np.array):
        """
        Computes the predicted values for a single observation.

        Args:
        - x_i: Index of the observation.
        - X: Input features.
        - thetas: Model parameters.

        Returns:
        - Predicted value for the observation.
        """
        guess = np.zeros(shape = (self.n_y_))
        priors = self.prior(X[x_i,:][np.newaxis,:], thetas)
        res = minimize(lambda y: -self.log_lik(X[x_i,:][np.newaxis,:], y, thetas, priors), 
                       guess, method = 'SLSQP', options={'disp': False})
        return res.x
    
    def __pred_func_naive(self, x_i: int, X: np.array, thetas: np.array):
        """
        Computes the predicted values for a single observation.

        Args:
        - x_i: Index of the observation.
        - X: Input features.
        - thetas: Model parameters.

        Returns:
        - Predicted value for the observation.
        """
        priors = self.prior(X[x_i,:][np.newaxis,:], thetas)
        gammas, betas, sigmas = self.unpack(thetas)
        pred_ys = X[x_i,:][np.newaxis,:].dot(betas.T)
        return pred_ys.dot(priors.T)[0,:,0]
            
    def predict_distr(self, X: np.array = None):
        """
        Predicts the distribution parameters for the given data.

        Args:
        - X: Input features. If None, use the training data.

        Returns:
        - Tuple containing predicted priors, values, and sigmas.
        """
        check_is_fitted(self)
        if X is None:
            X = self.X_
        else:
            X = self._validate_data(X, accept_sparse=False, reset=False)
            X = np.hstack((X, np.ones(shape=(X.shape[0], 1)))) if self.fit_intercept else X
        priors = self.prior(X, self.thetas_)
        gammas, betas, sigmas = self.unpack(self.thetas_)        
        return priors, X.dot(betas.T), sigmas
    
    def summary(self):
        """
        Prints a summary of the model parameters, log-likelihood, and BIC.
        """
        gammas, betas, sigmas = self.unpack(self.thetas_)
        gamma_stds, beta_stds, _ = self.unpack(np.sqrt(np.diag(self.var_thetas_)))
        var_cov_vecs = self.delta_method(self.thetas_, self.var_thetas_)
        sigma_stds = np.sqrt(var_cov_vecs)
        print('#'+'-'*91+'#')
        print('{:^10s}'.format('Model Parameters').center(93))
        print('#'+'-'*91+'#')
        print(' '*5+'Observation:'.ljust(20), '{}'.format(self.n_obs_).ljust(30), 'Success Flag:'.ljust(20), '{}'.format(self.flag_).ljust(20))
        print(' '*5+'Alpha:'.ljust(20), '{}'.format(self.alpha).ljust(30), 'LP Norm:'.ljust(20), '{}'.format(self.norm).ljust(20))
        print(' '*5+'Param Num:'.ljust(20), '{}'.format(len(self.thetas_)).ljust(30), 'DF:'.ljust(20), '{}'.format(self.n_obs_ - len(self.thetas_)).ljust(20))
        print(' '*5+'Log-lik:'.ljust(20), '{:.4f}'.format(self.log_lik_val_).ljust(30), 'BIC:'.ljust(20), '{:.4f}'.format(self.bic_val_).ljust(20))
        print('\n')
        print('Gamma: Logit Regression Coefficients')
        gammas = pd.DataFrame(gammas, index = ['state ' + str(x) for x in range(self.n_clusters-1)], columns = self.features_)
        gamma_stds = pd.DataFrame(gamma_stds, index = ['state '+str(x) for x in range(self.n_clusters-1)], columns = self.features_)
        gamma_pvals = pd.DataFrame(stats.norm.cdf(-np.abs(gammas.values/gamma_stds.values))*2, 
                                  index = ['state '+str(x) for x in range(self.n_clusters-1)], columns = self.features_)
        for index, row in gammas.iterrows():
            print('='*93)
            print('{:^10s}'.format(index).center(93))
            print('-'*93)
            print('|', 'vars'.center(20), '|',  
                  'gamma'.center(20), '|',  
                  'std err'.center(20), '|',  
                  'p value'.center(20), '|')
            print('-'*93)
            for col in gammas.columns:
                print('|', '{:^10s}'.format(col[:20]).center(20), '|', 
                      '{:.4f}'.format(row[col]).center(20), '|',  
                      '{:.4f}'.format(gamma_stds.loc[index, col]).center(20), '|',
                      '{:.4f}'.format(gamma_pvals.loc[index, col]).center(20), '|',
                      )
        print('='*93)
        print('\n')
        print('Beta: Main Model Regression Coefficients')
        for g in range(self.n_clusters):
            print('='*93)
            print('{:^10s}'.format('State ' + str(g)).center(93))
            betag = pd.DataFrame(betas[g,:,:].T, index = self.target_, columns = self.features_)
            beta_stdg = pd.DataFrame(beta_stds[g,:,:].T, index = self.target_, columns = self.features_)
            beta_pvalg = pd.DataFrame(stats.norm.cdf(-np.abs(betag.values/beta_stdg.values))*2, 
                                     index = self.target_, columns = self.features_)

            for index, row in betag.iterrows():
                print('-'*93)
                print('dependent var: {:^10s}'.format(index).center(93))
                print('-'*93)
                print('|', 'vars'.center(20), '|',  
                    'beta'.center(20), '|',  
                    'std err'.center(20), '|',  
                    'p value'.center(20), '|')
                print('-'*93)
                for col in betag.columns:
                    print('|', '{:^10s}'.format(col[:20]).center(20), '|', 
                        '{:.4f}'.format(row[col]).center(20), '|',  
                        '{:.4f}'.format(beta_stdg.loc[index, col]).center(20), '|',
                        '{:.4f}'.format(beta_pvalg.loc[index, col]).center(20), '|',
                        )
            print('='*93)
            print('\n')
        
        print('Sigma: Estimation Error Variance-Covariance Matrix')
        for g in range(self.n_clusters):
            print('='*93)
            print('{:^10s}'.format('State ' + str(g)).center(93))
            sigmag = pd.DataFrame(sigmas[g,:,:], index = self.target_, columns = self.target_)
            sigma_stdg = pd.DataFrame(sigma_stds[g,:,:], index = self.target_, columns = self.target_)
            sigma_pvalg = pd.DataFrame(stats.norm.cdf(-np.abs(sigmag.values/sigma_stdg.values))*2, index = self.target_, columns = self.target_)
            print('-'*93)
            print('|', 'vars-vars'.center(20), '|',  
                  'sigma2'.center(20), '|',  
                  'std err'.center(20), '|',  
                  'p value'.center(20), '|')
            for idyi in range(self.n_y_):
                for idyj in range(idyi+1):
                    print('-'*93)
                    sigmaid = self.target_[idyi][:8]+'-'+self.target_[idyj][:8]
                    print('|', '{:^10s}'.format(sigmaid).center(20), '|', 
                          '{:.4f}'.format(sigmag.iloc[idyi, idyj]).center(20), '|',  
                          '{:.4f}'.format(sigma_stdg.iloc[idyi, idyj]).center(20), '|',
                          '{:.4f}'.format(sigma_pvalg.iloc[idyi, idyj]).center(20), '|',)
            print('='*93)
            print('\n')

        print('#'+'-'*91+'#')
