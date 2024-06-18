"""
Markov Switching Gaussian Mixed Linear Regression for Sklearn toolkit API
"""

# Author: Zhenning Zhao
# Department of Economics
# University of Texas at Austin
# Last Update: 06/13/2024

import os
import time
from numba.np.unsafe import ndarray # essential to solve the bug in numba
import numpy as np
import pandas as pd
import tqdm
from typing import Literal
from numba import njit
import scipy.stats as stats
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils.validation import check_is_fitted
from helper.utils import Timer, gradient, dot3d


import warnings
warnings.filterwarnings('ignore')

@njit
def unpack_params(thetas: np.array, fit_cov: bool, fit_intercept:bool, n_clusters: int, n_X: int, n_y: int,
                 n_gammas: int, n_etas:int, n_betas:int, n_sigmas:int, n_covs:int):
    """
    Unpacks the flattened parameter array into individual matrices for further use.

    Parameters:
    - thetas: Numpy array containing the parameters of the model.
    - fit_cov: Boolean indicating if the covariance matrix should be fitted.
    - fit_intercept: Boolean indicating if the intercept should be fitted.
    - n_clusters: Number of clusters.
    - n_X: Number of features.
    - n_y: Number of response variables.
    - n_gammas: Number of gamma parameters.
    - n_etas: Number of eta parameters.
    - n_betas: Number of beta parameters.
    - n_sigmas: Number of sigma parameters.
    - n_covs: Number of covariance parameters.

    Returns:
    - Tuple of (gammas, etas, betas, sigmas), where each is a numpy array of parameters.
    """
    if fit_cov:
        assert len(thetas) == n_gammas + n_etas + n_betas + n_sigmas + n_covs
    else:
        assert len(thetas) == n_gammas + n_etas + n_betas + n_sigmas

    gammas = thetas[ : n_gammas]
    gammas = np.reshape(gammas, (n_clusters-1, n_X))
    etas = thetas[n_gammas : n_gammas+n_etas]
    etas = np.reshape(etas, (n_clusters-1, n_clusters-1 if fit_intercept else n_clusters))
    betas = thetas[n_gammas+n_etas : n_gammas+n_etas+n_betas]
    betas = np.reshape(betas, (n_clusters, n_X, n_y))
    sigmas = thetas[n_gammas + n_etas + n_betas : n_gammas + n_etas + n_betas + n_sigmas]
    sigmas = np.reshape(sigmas, (n_clusters, n_y))
    sigma_list = [np.array(np.diag(sigmas[g])) for g in range(n_clusters)]
    sigmas = np.zeros((n_clusters, n_y, n_y))
    for g in range(n_clusters):
        sigmas[g,:,:] = sigma_list[g]

    if fit_cov and int(n_y*(n_y-1)/2)>0:
        # lower trianglular decomposition factors
        covs = thetas[n_gammas + n_etas + n_betas + n_sigmas : n_gammas + n_etas + n_betas + n_sigmas + n_covs]
        covs = np.reshape(covs, (n_clusters, int(n_y*(n_y-1)/2)))
        for g in range(n_clusters):
            k = 0
            for i in range(1, n_y):
                for j in range(0, i):
                    sigmas[g, i, j] = covs[g, k]
                    k += 1
    
    for g in range(n_clusters):
        sigmas[g] = sigmas[g].dot(sigmas[g].T)
        
    return gammas, etas, betas, sigmas

@njit
def update(X: np.array, y: np.array, thetas: np.array, fit_cov: bool, fit_intercept:bool, n_obs: int, n_clusters: int, 
           n_X: int, n_y: int, n_gammas: int, n_etas:int, n_betas:int, n_sigmas:int, n_covs:int, init_guess = None):
    """
    Updates the model parameters using Bayes Rule to calculate priors and posteriors for each state.

    Parameters:
    - X: Numpy array of predictors.
    - y: Numpy array of response variables.
    - thetas: Numpy array of model parameters.
    - fit_cov: Boolean indicating if the covariance matrix should be fitted.
    - fit_intercept: Boolean indicating if the intercept should be fitted.
    - n_obs: Number of observations.
    - n_clusters: Number of clusters.
    - n_X: Number of features.
    - n_y: Number of response variables.
    - n_gammas: Number of gamma parameters.
    - n_etas: Number of eta parameters.
    - n_betas: Number of beta parameters.
    - n_sigmas: Number of sigma parameters.
    - n_covs: Number of covariance parameters.
    - init_guess: Initial guess for the prior probabilities.

    Returns:
    - Tuple of (priors, posteriors), where each is a numpy array of calculated probabilities.
    """
    # use Bayes Rule to calculate the prior and the postior simultaneously for each state
    if init_guess is None:
        init_guess = np.zeros(shape=(1, n_clusters - 1 if fit_intercept else n_clusters))
    gammas, etas, betas, sigmas = unpack_params(thetas, fit_cov, fit_intercept, n_clusters, 
                                                n_X, n_y, n_gammas, n_etas, n_betas, n_sigmas, n_covs)
    y = y.reshape(X.shape[0], n_y)
    ymtx = np.zeros((X.shape[0], n_y, n_clusters))
    for g in range(n_clusters):
        ymtx[:,:,g] = y
    err = ymtx - dot3d(X, betas)

    lagP = init_guess
    priors = np.zeros((n_obs, n_clusters))
    postiors = np.zeros((n_obs, n_clusters))
    for t in range(n_obs):
        denominator = 1 + np.sum(np.exp(X[t,:].dot(gammas.T) + lagP.dot(etas.T)))
        nominator = np.hstack((np.exp(X[t,:].dot(gammas.T) + lagP.dot(etas.T)), np.ones((1, 1))))
        prior = nominator / denominator
        priors[t, :] = prior
        pmtx = np.zeros((1, n_clusters))
        for g in range(n_clusters):
            # Probability that observation falls in the i-th group
            curr_err = err[t,:,g][np.newaxis,:]
            try:
                normalprob = np.exp(-1.0/2.0 * np.diag(curr_err.dot(np.linalg.inv(sigmas[g])).dot(curr_err.T)))
            except:
                Warning("Singular matrix. Trying Penrose inverse.")
                normalprob = np.exp(-1.0/2.0 * np.diag(curr_err.dot(np.linalg.pinv(sigmas[g])).dot(curr_err.T)))
                
            pval = prior[0, g] / np.sqrt(2.0*np.pi*np.linalg.det(sigmas[g])) * normalprob # ??????
            pmtx[:, g] = pval
        postior = pmtx / np.sum(pmtx, axis=1)
        postiors[t, :] = postior
        lagP = postior[:, :-1] if fit_intercept else postior
    return priors, postiors

class SklearnMSGMLR(BaseEstimator):
    """
    A Markov Switching Gaussian Mixture Linear Regression (MS-GMLR) model compatible with scikit-learn.

    This model fits a Markov Switching Gaussian mixture model to the data, allowing for 
    a flexible representation of the underlying distributions.

    Parameters
    ----------
    ascending: bool, default=True
        Whether get ascending data order.

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
    >>> from gmlr.api.gmlr import SklearnMSGMLR
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> msgmlr_mod = SklearnMSGMLR()
    >>> msgmlr_mod.fit(X, y)
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "ascending": [bool],
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

    def __init__(self, ascending: bool = True, n_clusters: int = 2, fit_intercept: bool = True, fit_cov: bool = False, alpha: float = 0.0, 
                 norm: int|float = 1, warm_start: bool = False, path: str = None, max_iter:int = 100,
                 tol:float = 1e-4, step_plot: callable = None, pred_mode: Literal['naive', 'loglik'] = 'naive', verbose = 0):
        self.ascending = ascending
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
        self.n_etas_ = (self.n_clusters-1)**2 if self.fit_intercept else self.n_clusters*(self.n_clusters-1)
        self.n_betas_ = self.n_X_*self.n_clusters*self.n_y_
        self.n_sigmas_ = self.n_clusters*self.n_y_
        self.n_covs_ = self.n_clusters*int(self.n_y_*(self.n_y_-1)/2)
        self.slices_ = dict()
        self.slices_['gammas'] = np.index_exp[:self.n_gammas_]
        self.slices_['etas']   = np.index_exp[self.n_gammas_: 
                                              self.n_gammas_ + self.n_etas_]
        self.slices_['betas']  = np.index_exp[self.n_gammas_ + self.n_etas_: 
                                              self.n_gammas_ + self.n_etas_ + self.n_betas_]
        self.slices_['sigmas'] = np.index_exp[self.n_gammas_ + self.n_etas_ + self.n_betas_: 
                                              self.n_gammas_ + self.n_etas_ + self.n_betas_ + self.n_sigmas_]
        self.slices_['covs']   = np.index_exp[self.n_gammas_ + self.n_etas_ + self.n_betas_ + self.n_sigmas_: 
                                              self.n_gammas_ + self.n_etas_ + self.n_betas_ + self.n_sigmas_ + self.n_covs_]
    
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
        """
        return unpack_params(thetas, self.fit_cov, self.fit_intercept, self.n_clusters, self.n_X_, self.n_y_, 
                             self.n_gammas_, self.n_etas_, self.n_betas_, self.n_sigmas_, self.n_covs_)
    
    def update(self, X: np.array, y: np.array, thetas: np.array, init_guess = None):
        n_obs = X.shape[0]
        return update(X, y, thetas, self.fit_cov, self.fit_intercept, 
                      n_obs, self.n_clusters, self.n_X_, self.n_y_, self.n_gammas_, 
                      self.n_etas_, self.n_betas_, self.n_sigmas_, self.n_covs_, init_guess)
    
    def partial_lik(self, X: np.array, y: np.array, thetas: np.array, init_guess = None):
        """
        Parameters:
        - X: Numpy array of independent variables.
        - y: Numpy array of dependent variables.
        - thetas: Numpy array containing the parameters of the model.
        - init_guess: Initial guess for the model parameters.

        Returns:
        - Value of the partial likelihood function.
        """
        gammas, etas, betas, sigmas = self.unpack(thetas)
        priors, postiors = self.update(X, y, thetas, init_guess)
        y = y.reshape(X.shape[0], self.n_y_)
        ymtx = np.repeat(y[:, :, np.newaxis], self.n_clusters, axis=2)
        err = ymtx - X.dot(betas.T)
        
        pmtx = []
        for g in range(self.n_clusters):
            # Probability that observation falls in the i-th group
            normal_prob = np.exp(-1.0/2.0 * np.diag(err[:,:,g].dot(np.linalg.inv(sigmas[g])).dot(err[:,:,g].T)))
            pval = priors[:,g] / np.sqrt(2.0*np.pi*np.linalg.det(sigmas[g])) * normal_prob
            pmtx.append(pval)
        return np.array(pmtx).T
    
    def filter(self, X: np.array, y: np.array, thetas: np.array, priors: np.array, postiors: np.array):
        gammas, etas, betas, sigmas = self.unpack(thetas)
        n_obs = X.shape[0]
        probs = np.zeros(shape=(n_obs, self.n_clusters))
        curr_smoothed_prob = probs[n_obs-1,:] = postiors[-1,:]
        for t in range(n_obs-1):
            pt = n_obs-t-1
            lagPs = np.zeros((self.n_clusters, self.n_clusters-1)) if self.fit_intercept else np.zeros((self.n_clusters, self.n_clusters))
            for g in range(self.n_clusters-1 if self.fit_intercept else self.n_clusters):
                lagPs[g, g] = 1
            wedge = np.zeros(self.n_clusters)
            Ptrans = np.zeros((self.n_clusters, self.n_clusters))
            for g in range(self.n_clusters):
                lagP = lagPs[g, :]
                denominator = 1 + np.sum(np.exp(X[pt,:].dot(gammas.T) + lagP.dot(etas.T)))
                nominator = np.hstack((np.exp(X[pt,:].dot(gammas.T) + lagP.dot(etas.T)), np.ones((1))))
                Ptrans[g,:] = nominator / denominator  # used in the nominator
            for g in range(self.n_clusters):
                bottom = np.array([sum(Ptrans[:,x] * postiors[pt-1, :]) for x in range(self.n_clusters)])
                wedge[g] = np.sum(Ptrans[g] * curr_smoothed_prob / bottom)
            curr_smoothed_prob = probs[pt-1] = wedge * postiors[pt-1, :]
        return probs
    
    def log_lik(self, X: np.array, y: np.array, thetas: np.array, postiors: np.array, init_guess = None):
        """
        Computes the log-likelihood function.

        Parameters:
        - X: Numpy array of independent variables.
        - y: Numpy array of dependent variables.
        - thetas: Numpy array containing the parameters of the model.
        - postiors: Numpy array containing posterior probabilities.

        Returns:
        - Log-likelihood value.
        """
        # take expectation of the log-likilihood function
        loglik = np.log(self.partial_lik(X, y, thetas, init_guess))
        loglik = np.sum(postiors * loglik)
        return loglik
    
    def penalty(self, thetas: np.array):
        # Computes the penalty for regularization.
        return np.nansum(abs(thetas)**self.norm)**(1/self.norm)

    def guess(self):
        # Generate an initial guess for the model parameters
        # initialize the first guess as the OLS regression beta using only a slice of the data
        def gen_guess():
            if self.fit_cov:
                guess = np.zeros(self.n_gammas_ + self.n_etas_ + self.n_betas_ + self.n_sigmas_ + self.n_covs_)
            else:
                guess = np.zeros(self.n_gammas_ + self.n_etas_ + self.n_betas_ + self.n_sigmas_)

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
                    assert len(guess) == self.n_gammas_ + self.n_etas_ + self.n_betas_ + self.n_sigmas_ + self.n_covs_
                else:
                    assert len(guess) == self.n_gammas_ + self.n_etas_ + self.n_betas_ + self.n_sigmas_
            except:
                guess = gen_guess()
        return guess

    def model_fit(self, plot: callable):
        """
        Fits the model using the EM algorithm.

        Parameters:
        - plot: Callable function to plot the model fitting process.
        
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
            priors, postiors = self.update(self.X_, self.y_, thetas)
            smoothed = self.filter(self.X_, self.y_, thetas, priors, postiors)
            res = minimize(lambda thetas: -self.log_lik(self.X_, self.y_, thetas, smoothed) + self.alpha * self.penalty(thetas), 
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
                print('Warning: maximum number of iteration reached.')
                flag = False
        log_lik_val = self.log_lik(self.X_, self.y_, thetas, postiors)
        end = time.time()
        if self.verbose > 0:
            print('EM Estimation Completed in {:.4f} seconds.'.format(end-start)) 
        return thetas, log_lik_val, flag

    def hessian(self, X: np.array, y: np.array, thetas: np.array, probs: np.array):
        """
        Compute the Hessian matrix.

        Parameters:
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

        Parameters:
        - log_lik: Log-likelihood value.
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - BIC value.
        """
        k_theta = len(thetas)
        log_n = np.log(self.n_obs_)
        bic_val = k_theta*log_n - 2*log_lik
        return bic_val

    def delta_method(self, thetas: np.array, var_thetas: np.array):
        """
        Computes the variance-covariance matrix using the delta method.

        Parameters:
        - thetas: Numpy array containing the parameters of the model.
        - var_thetas: 2-D Numpy array containing the variance-covariance matrix of the model parameters.

        Returns:
        - Variance-covariance matrix as a numpy array.
        """
        delta = gradient(thetas, lambda thetas: self.unpack(thetas)[-1])   ### ?
        if self.fit_cov:
            delta = np.concatenate([delta[self.slices_['sigmas']], delta[self.slices_['covs']]], axis = 0)  
        else:
            delta = delta[self.slices_['sigmas']]
        
        varcov_vecs = []
        for g in range(self.n_clusters):
            delta_g = delta[:, g, :, :].reshape(delta.shape[0], self.n_y_*self.n_y_)
            varmtx = var_thetas[-delta.shape[0]:, -delta.shape[0]:]
            varcov_vec = np.diag(delta_g.T.dot(varmtx).dot(delta_g)).reshape(self.n_y_, self.n_y_)
            varcov_vecs.append(varcov_vec)
        return np.array(varcov_vecs)

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
        if not self.ascending:
            X, y =  X[::-1], y[::-1]
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
        self.priors_, self.postiors_ = self.update(self.X_, self.y_, self.thetas_)
        self.smoothed_ = self.filter(self.X_, self.y_, self.thetas_, self.priors_, self.postiors_)
        hex = self.hessian(self.X_, self.y_, self.thetas_, self.smoothed_)
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
        y = []
        if pred_std: 
            stds = []
        if self.verbose > 0:
            print('Model Prediction Started...')

        if X is None:
            X = self.X_
            with Timer("In Sample Prediction", display=self.verbose > 0):
                priors, postiors = self.update(self.X_, self.y_, self.thetas_)
                smoothed = self.filter(self.X_, self.y_, self.thetas_, priors, postiors)
                for xi in tqdm.trange(X.shape[0], disable = self.verbose < 2, desc = 'Model Prediction'):
                    init_guess = (smoothed[xi-1,:][np.newaxis,:-1] if self.fit_intercept else smoothed[xi-1,:][np.newaxis,:]) if xi>0 else None
                    if self.pred_mode == 'naive':
                        pred_func = lambda thetas: self.__pred_func_naive(X[xi,:][np.newaxis,:], priors[xi,:][np.newaxis,:], thetas)
                    else:
                        pred_func = lambda thetas: self.__pred_func(X[xi,:][np.newaxis,:], priors[xi,:][np.newaxis,:], thetas, init_guess)
                    yi = pred_func(self.thetas_)
                    y.append(yi)
                    if pred_std:
                        grad_yi = gradient(self.thetas_, pred_func) 
                        pred_vars = grad_yi.T.dot(self.var_thetas_).dot(grad_yi)
                        stds.append(np.sqrt(np.diag(pred_vars)))
        else:
            X = self._validate_data(X, accept_sparse=False, reset=False)
            X = np.hstack((X, np.ones(shape=(X.shape[0], 1)))) if self.fit_intercept else X
            with Timer('Out of Sample Model Prediction', display=self.verbose > 0):
                priors, postiors = self.update(self.X_, self.y_, self.thetas_)
                smoothed = self.filter(self.X_, self.y_, self.thetas_, priors, postiors)
                curr_prob = smoothed[-1,:][np.newaxis,:-1] if self.fit_intercept else smoothed[-1,:][np.newaxis,:]
                for xi in tqdm.trange(X.shape[0], disable = self.verbose < 2, desc = 'Model Prediction', leave = False):
                    init_guess = curr_prob
                    # calculate the probability used in calculating the out-of-sample prediction
                    gammas, etas, betas, sigmas = self.unpack(self.thetas_)
                    denominator = 1 + np.sum(np.exp(X[xi,:][np.newaxis,:].dot(gammas.T) + init_guess.dot(etas.T)))
                    nominator = np.hstack((np.exp(X[xi,:][np.newaxis,:].dot(gammas.T) + init_guess.dot(etas.T)), np.ones((1, 1))))
                    prior = nominator / denominator
                    # do out-of-sample prediction
                    if self.pred_mode == 'naive':
                        pred_func = lambda thetas: self.__pred_func_naive(X[xi,:][np.newaxis,:], prior, thetas)
                    else:
                        pred_func = lambda thetas: self.__pred_func_oos(X[xi,:][np.newaxis,:], prior, thetas, init_guess)
                    yi = pred_func(self.thetas_)
                    y.append(yi)
                    if pred_std:
                        grad_yi = gradient(self.thetas_, pred_func) 
                        pred_vars = grad_yi.T.dot(self.var_thetas_).dot(grad_yi)
                        stds.append(np.sqrt(np.diag(pred_vars)))
                    priors, postiors = self.update(X[xi,:][np.newaxis,:], yi[np.newaxis,:], self.thetas_, init_guess)
                    smoothed = self.filter(X[xi,:][np.newaxis,:], yi[np.newaxis,:], self.thetas_, priors, postiors)
                    curr_prob = smoothed[-1,:][np.newaxis,:-1] if self.fit_intercept else smoothed[xi-1,:][np.newaxis,:]
        
        preds = np.array(y)
        if pred_std:
            stds = np.array(stds)
            return preds, stds
        else:
            return preds
    
    def __pred_func(self, X: np.array, smoothed: np.array, thetas: np.array, init_guess = None):
        """
        Private method to perform prediction using the fitted model parameters.

        Parameters
        ----------
        X : np.array
            Input data for prediction.
        smoothed : np.array
            Smoothed state estimates from the model.
        thetas : np.array
            Model parameters.
        init_guess : np.array or None, optional
            Initial guess for the prediction, default is None.

        Returns
        -------
        np.array
            Predicted values.
        """
        guess = np.zeros(shape = (1, self.n_y_))
        res = minimize(lambda y: -self.log_lik(X, y, thetas, smoothed, init_guess), 
                       guess, method = 'SLSQP', options={'disp': False})
        return res.x
    
    def __pred_func_oos(self, X: np.array, postiors: np.array, thetas: np.array, init_guess = None):
        """
        Private method to perform out-of-sample prediction using the fitted model parameters.

        Parameters
        ----------
        X : np.array
            Input data for prediction.
        posteriors : np.array
            Posterior state estimates from the model.
        thetas : np.array
            Model parameters.
        init_guess : np.array or None, optional
            Initial guess for the prediction, default is None.

        Returns
        -------
        np.array
            Predicted values.
        """
        guess = np.zeros(shape = (1, self.n_y_))
        res = minimize(lambda y: -self.log_lik(X, y, thetas, postiors, init_guess), 
                       guess, method = 'SLSQP', options={'disp': False})
        return res.x
    
    def __pred_func_naive(self, X: np.array, priors: np.array, thetas: np.array):
        """
        Private method to perform naive prediction using the fitted model parameters.

        Parameters
        ----------
        X : np.array
            Input data for prediction.
        priors : np.array
            Prior state estimates from the model.
        thetas : np.array
            Model parameters.

        Returns
        -------
        np.array
            Predicted values.
        """
        gammas, etas, betas, sigmas = self.unpack(thetas)
        pred_ys = X.dot(betas.T)
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
            priors, postiors = self.update(self.X_, self.y_, self.thetas_)
        else:
            X = self._validate_data(X, accept_sparse=False, reset=False)
            y = self.predict(X)
            X = np.hstack((X, np.ones(shape=(X.shape[0], 1)))) if self.fit_intercept else X
            priors, postiors = self.update(self.X_, self.y_, self.thetas_)
            smoothed = self.filter(self.X_, self.y_, self.thetas_, priors, postiors)
            curr_prob = smoothed[-1,:][np.newaxis,:-1] if self.fit_intercept else smoothed[-1,:][np.newaxis,:]
            priors, postiors = self.update(X, y, self.thetas_, curr_prob)
        gammas, etas, betas, sigmas = self.unpack(self.thetas_)

        return priors, X.dot(betas.T), sigmas
    
    def summary(self):
        """
        Prints a summary of the model parameters, log-likelihood, and BIC.
        """
        gammas, etas, betas, sigmas = self.unpack(self.thetas_)
        gamma_stds, eta_stds, beta_stds, _ = self.unpack(np.sqrt(np.diag(self.var_thetas_)))
        varcov_vecs = self.delta_method(self.thetas_, self.var_thetas_)
        sigma_stds = np.sqrt(varcov_vecs)
        print('#'+'-'*91+'#')
        print('{:^10s}'.format('Model Parameters').center(93))
        print('#'+'-'*91+'#')
        print(' '*5+'Observation:'.ljust(20), '{}'.format(self.n_obs_).ljust(30), 'Success Flag:'.ljust(20), '{}'.format(self.flag_).ljust(20))
        print(' '*5+'Alpha:'.ljust(20), '{}'.format(self.alpha).ljust(30), 'LP Norm:'.ljust(20), '{}'.format(self.norm).ljust(20))
        print(' '*5+'Param Num:'.ljust(20), '{}'.format(len(self.thetas_)).ljust(30), 'DF:'.ljust(20), '{}'.format(self.n_obs_ - len(self.thetas_)).ljust(20))
        print(' '*5+'Log-lik:'.ljust(20), '{:.4f}'.format(self.log_lik_val_).ljust(30), 'BIC:'.ljust(20), '{:.4f}'.format(self.bic_val_).ljust(20))
        print('\n')
        print('Gamma & Eta: Logit Regression Coefficients')
        gammas = pd.DataFrame(gammas, index = ['state '+str(x) for x in range(self.n_clusters-1)], columns = self.features_)
        gamma_stds = pd.DataFrame(gamma_stds, index = ['state '+str(x) for x in range(self.n_clusters-1)], columns = self.features_)
        gamma_pvals = pd.DataFrame(stats.norm.cdf(-np.abs(gammas.values/gamma_stds.values))*2, 
                                  index = ['state '+str(x) for x in range(self.n_clusters-1)], columns = self.features_)
        eta_cols = ['P(L.state '+str(x)+')' for x in range(self.n_clusters-1 if self.fit_intercept else self.n_clusters)]
        etas = pd.DataFrame(etas, index = ['state '+str(x) for x in range(self.n_clusters-1)],
                            columns = eta_cols)
        eta_stds = pd.DataFrame(eta_stds, index = ['state '+str(x) for x in range(self.n_clusters-1)], 
                               columns = eta_cols)
        eta_pvals = pd.DataFrame(stats.norm.cdf(-np.abs(etas.values/eta_stds.values))*2, 
                                index = ['state '+str(x) for x in range(self.n_clusters-1)], 
                                columns = eta_cols)

        for index, row in gammas.iterrows():
            print('='*93)
            print('{:^10s}'.format(index).center(93))
            print('-'*93)
            print('|', 'vars'.center(20), '|',  
                  'gamma'.center(20), '|',  
                  'std err'.center(20), '|',  
                  'p value'.center(20), '|')
            print('-'*93)
            Xcol = self.features_ if self.fit_intercept else self.features_[:-1]
            for col in Xcol:
                print('|', '{:^10s}'.format(col[:20]).center(20), '|', 
                      '{:.4f}'.format(row[col]).center(20), '|',  
                      '{:.4f}'.format(gamma_stds.loc[index, col]).center(20), '|',
                      '{:.4f}'.format(gamma_pvals.loc[index, col]).center(20), '|',
                      )
            for col in eta_cols:
                print('|', '{:^10s}'.format(col[:20]).center(20), '|', 
                      '{:.4f}'.format(etas.loc[index, col]).center(20), '|',  
                      '{:.4f}'.format(eta_stds.loc[index, col]).center(20), '|',
                      '{:.4f}'.format(eta_pvals.loc[index, col]).center(20), '|',
                      )
            if self.fit_intercept:
                col = 'Const'
                print('|', '{:^10s}'.format(col[:20]).center(20), '|', 
                    '{:.4f}'.format(gammas.loc[index, col]).center(20), '|',  
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
