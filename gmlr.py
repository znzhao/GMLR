import time
import copy
import warnings
import tqdm
import pandas as pd
import numpy as np
from typing import Literal
from matplotlib import pyplot
from scipy.stats import norm, multivariate_normal
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from IPython.display import clear_output
from IPython import get_ipython
from helper.utils import Timer, gradient, colormap
warnings.filterwarnings('ignore')

class CombinedDistr:
    def __init__(self, priors, means, covs):
        self.ngroup = means.shape[-1]
        self.priors = priors
        self.means = means
        self.covs = covs

    def pdf(self, x):
        res = 0
        for g in range(self.ngroup):
            res += self.priors[g] * multivariate_normal.pdf(x, mean=self.means[:, g], cov=self.covs[g, :, :])
        return res

    def margpdf(self, x, margin = 0):
        res = 0
        for g in range(self.ngroup):
            res += self.priors[g] * multivariate_normal.pdf(x, mean=[self.means[margin, g]], cov=[self.covs[g, margin, margin]])
        return res
    
    def margcdf(self, x, margin = 0):
        res = 0
        for g in range(self.ngroup):
            res += self.priors[g] * multivariate_normal.cdf(x, mean=[self.means[margin, g]], cov=[self.covs[g, margin, margin]])
        return res
    
    def plot(self, margin = 0, ax = None, figsize = (14,8), show = True):
        if ax is None:
            fig, ax = pyplot.subplots(1, 2, figsize = figsize, width_ratios=[1, 3])
        ax[0].bar(x = ['state {}'.format(g) for g in range(self.ngroup)],height = self.priors, color = [colormap[g] for g in range(self.ngroup)])
        ax[0].axis('tight')
        [ax[0].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax[0].tick_params(axis='both', which='major', labelsize=10)
        ax[0].set_ylabel('Probability', fontsize=12)
        ax[0].set_title('State Probability', fontsize=14)
        
        std = np.sum([np.sum(np.sqrt(np.diag(self.covs[g,:,:]))) for g in range(self.ngroup)])
        x = np.linspace(np.min(self.means[margin, :]) - std, np.max(self.means[margin, :]) + std, 200)
        for g in range(self.ngroup):
            ax[1].axvspan(self.means[margin, g] - 1.68*np.sqrt(self.covs[g, margin, margin]), 
                    self.means[margin, g] + 1.68*np.sqrt(self.covs[g, margin, margin]), 
                    alpha=0.4, color=colormap[g])
        ax[1].plot(x, self.margpdf(x, margin), "grey")
        ax[1].scatter(x = self.means[margin, :], y = self.margpdf(self.means[margin, :], margin), marker = 'o', color = [colormap[g] for g in range(self.ngroup)])
        ax[1].axvline(x = 0, color="black", linestyle = '--')
        ax[1].axhline(y = 0, color="black", linestyle = '-')
        ax[1].vlines(self.means[margin, :], 0, self.margpdf(self.means[margin, :], margin), linestyle="dashed", color = [colormap[g] for g in range(self.ngroup)])
        ax[1].hlines(self.margpdf(self.means[margin, :], margin), 0, self.means[margin, :], linestyle="dashed", color = [colormap[g] for g in range(self.ngroup)])
        ax[1].axis('tight')
        [ax[1].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax[1].tick_params(axis='both', which='major', labelsize=10)
        ax[1].set_xlabel('Predicted Value', fontsize=12)
        ax[1].set_ylabel('Prob Density', fontsize=12)
        ax[1].set_title('Prob Density Function P(Value<0) = {:.2f}'.format(self.margcdf(0, margin)), fontsize=14)
        if show: pyplot.show()

class GMLR:
    def __init__(self, data: pd.DataFrame, ycol: list, Xcol: list, ngroup = 2, const = True, cov = False, alpha = 0, costnorm = 1):
        """
        Initialize the Gaussian Mixture Linear Regression (GMLR) model with adjustable state-dependent probability.

        Parameters:
        - data: Pandas DataFrame containing the dataset.
        - ycol: List of column names for the dependent variable(s).
        - Xcol: List of column names for the independent variable(s).
        - ngroup: Number of groups or states in the model.
        - const: Boolean indicating whether to include a constant term in the regression model.
        - cov: Boolean indicating whether to include a covariance term in the variance-covariance matrix.
        ---------------------------------------------------------------------------------------------
        The GMLR class is designed for fitting and predicting a mixture model using the Expectation-Maximization (EM) algorithm.
        To initialize the class, create an instance by providing a dataset in the form of a Pandas DataFrame (`data`), specifying the
        column names for the dependent variables (`ycol`), independent variables (`Xcol`), the number of groups or states (`ngroup`),
        and whether to include a constant term in the regression model (`const`). The instance is initialized with these parameters.
        Once initialized, the `fit` method can be called to train the model using the EM algorithm. It takes optional parameters such as
        `maxiter` (maximum number of iterations), `tol` (convergence tolerance), `disp` (display progress), `plot` (plotting the model
        during training), `plotx` (x-axis variable for plotting), and `ploty` (y-axis variable for plotting). After fitting, the
        `predict` method can be used to generate predictions for the dependent variable. If no independent variables are provided, it
        uses the training data by default. Additionally, the `summary` method can be called to print a summary of the model parameters,
        log-likelihood, and Bayesian Information Criterion (BIC). Below is an example of using this class:

        Example Usage:
        ```
        import pandas as pd
        import numpy as np
        from EMclassifier import EMclassifier
        
        # Create a synthetic dataset
        np.random.seed(42)
        data = pd.DataFrame({
            'X1': np.random.randn(100),
            'X2': np.random.randn(100),
            'y1': np.random.choice([0, 1], size=100),
            'y2': np.random.choice([0, 1], size=100)
        })
        data['z'] = np.random.choice([0, 1], size=100)

        # Initialize the EMclassifier instance
        em_model = EMclassifier(data, ycol=['y1', 'y2'], Xcol=['X1', 'X2'], ngroup=2, const=True)

        # Fit the model using the EM algorithm
        estimated_parameters = em_model.fit(maxiter=50, tol=1e-6, disp=True, plot=True, plotx='X1', ploty='y1')

        # Generate predictions for the dependent variables
        predictions = em_model.predict()

        # Print a summary of the model parameters, log-likelihood, and BIC
        em_model.summary()
        ```
        """
        self.data = data
        self.nobs = data.shape[0]
        self.y = data[ycol].values
        self.ny = len(ycol)
        self.ycol = ycol
        self.const = const
        self.X = np.hstack((data[Xcol].values, np.ones(shape=(self.nobs, 1)))) if const else data[Xcol].values
        self.nX = self.X.shape[1]
        self.Xcol = Xcol
        self.ngroup = ngroup
        self.ngammas = (self.ngroup-1)*self.nX
        self.nbetas = self.nX*self.ngroup*self.ny
        self.nsigmas = self.ngroup*self.ny
        self.ncovs = self.ngroup*int(self.ny*(self.ny-1)/2)
        
        self.slices = dict()
        self.slices['gammas'] = np.index_exp[:self.ngammas]
        self.slices['betas'] = np.index_exp[self.ngammas: self.ngammas + self.nbetas]
        self.slices['sigmas'] = np.index_exp[self.ngammas + self.nbetas: self.ngammas + self.nbetas + self.nsigmas]
        self.slices['covs'] = np.index_exp[self.ngammas + self.nbetas + self.nsigmas: self.ngammas + self.nbetas + self.nsigmas + self.ncovs]
        
        self.cov = cov
        self.alpha = alpha
        self.norm = costnorm


    def __unpack(self, thetas: np.array):
        """
        Unpacks the flattened parameter array into individual matrices for further use.

        Parameters:
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - Tuple of gammas, betas, and sigmas (parameters of the model).
        """
        if self.cov:
            assert len(thetas) == self.ngammas + self.nbetas + self.nsigmas + self.ncovs
        else:
            assert len(thetas) == self.ngammas + self.nbetas + self.nsigmas

        gammas = thetas[self.slices['gammas']]
        gammas = np.reshape(gammas, newshape=(self.ngroup-1, self.nX))
        betas = thetas[self.slices['betas']]
        betas = np.reshape(betas, newshape=(self.ngroup, self.nX, self.ny))
        sigmas = thetas[self.slices['sigmas']]
        sigmas = np.reshape(sigmas, newshape=(self.ngroup, self.ny))
        sigmas = np.array([np.diag(sigmas[g]) for g in range(self.ngroup)])

        if self.cov and int(self.ny*(self.ny-1)/2)>0:
            # lower trianglular decomposition factors
            covs = thetas[self.slices['covs']]
            covs = np.reshape(covs, newshape=(self.ngroup, int(self.ny*(self.ny-1)/2)))
            for g in range(self.ngroup):
                k = 0
                for i in range(1, self.ny):
                    for j in range(0, i):
                        sigmas[g, i, j] = covs[g, k]
                        k += 1
        
        for g in range(self.ngroup):
            sigmas[g] = sigmas[g].dot(sigmas[g].T)
            
        return gammas, betas, sigmas
    
    def unpack(self, thetas: np.array):
        """
        Public interface to unpack the model parameters.

        Parameters:
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - Tuple of gammas, betas, and sigmas (parameters of the model).
        """
        return self.__unpack(thetas)
    
    def prior(self, X: np.array, thetas: np.array):
        """
        Computes the prior probabilities used in both E-step and M-step.

        Parameters:
        - X: Numpy array of independent variables.
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - Matrix of prior probabilities for each observation belonging to each group.
        """
        gammas, betas, sigmas = self.__unpack(thetas)
        # initialize pmtx
        logit_denominator = 1 + np.sum(np.exp(X.dot(gammas.T)), axis = 1)
        logit_denominator = np.repeat(logit_denominator[:, np.newaxis], repeats = self.ngroup, axis = 1)
        logit_nominator = np.hstack((np.exp(X.dot(gammas.T)), np.ones((X.shape[0], 1))))
        logit = logit_nominator / logit_denominator
        return logit
    
    def partialLik(self, X: np.array, y: np.array, thetas: np.array):
        """
        Computes the partial likelihood function for the model.

        Parameters:
        - X: Numpy array of independent variables.
        - y: Numpy array of dependent variables.
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - Value of the partial likelihood function.
        """
        gammas, betas, sigmas = self.__unpack(thetas)
        logit = self.prior(X, thetas)
        y = y.reshape(X.shape[0], self.ny)
        ymtx = np.repeat(y[:, :, np.newaxis], self.ngroup, axis=2)
        err = ymtx - X.dot(betas.T)
        
        pmtx = []
        for g in range(self.ngroup):
            # Probability that observation falls in the i-th group
            normalprob = np.exp(-1.0/2.0 * np.diag(err[:,:,g].dot(np.linalg.inv(sigmas[g])).dot(err[:,:,g].T)))
            pval = logit[:,g] / np.sqrt(2.0*np.pi*np.linalg.det(sigmas[g])) * normalprob
            pmtx.append(pval)
        return np.array(pmtx).T

    def postior(self, X: np.array, y: np.array, thetas: np.array):
        """
        Implements the expectation step in the EM algorithm.

        Parameters:
        - X: Numpy array of independent variables.
        - y: Numpy array of dependent variables.
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - Matrix of updated probabilities according to Bayesian Rule.
        """
        # expectation step: for each observation, calculate the probability to fall in group G
        prob = self.partialLik(X, y, thetas)
        # update probability according to Bayesian Rule
        return prob / np.repeat(prob.sum(axis=1)[:, np.newaxis], self.ngroup, axis=1)
    
    def logLik(self, X: np.array, y: np.array, thetas: np.array, probs: np.array):
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
        loglik = np.sum(probs * np.log(self.partialLik(X, y, thetas)))
        return loglik

    def penalty(self, thetas: np.array):
        return np.nansum(abs(thetas)**self.norm)**(1/self.norm)
    
    def modelFit(self, inputX, inputy, maxiter = 100, tol = 1e-4, disp = True, plot = True, plotx = None, ploty = None):
        """
        Fits the model using the EM algorithm.

        Args:
        - inputX: Input features.
        - inputy: Target labels.
        - maxiter: Maximum number of iterations for the EM algorithm.
        - tol: Tolerance for convergence.
        - disp: Boolean indicating whether to display progress.
        - plot: Boolean indicating whether to plot the model during training.
        - plotx: Name of the x-axis variable for plotting.
        - ploty: Name of the y-axis variable for plotting.

        Returns:
        - Numpy array of final estimated parameters.
        """
        # initialize the first guess as the OLS regression beta using only a slice of the data
        if self.cov:
            guess = np.zeros(self.ngammas + self.nbetas + self.nsigmas + self.ncovs)
        else:
            guess = np.zeros(self.ngammas + self.nbetas + self.nsigmas)

        betas = []
        covs = []
        for g in range(self.ngroup):
            y = inputy[g*int(self.nobs/self.ngroup): (g+1)*int(self.nobs/self.ngroup)]            
            X = inputX[g*int(self.nobs/self.ngroup): (g+1)*int(self.nobs/self.ngroup)]
            cov = np.cov(y.T)
            beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
            betas.append(beta)
            covs.append(cov)
        betas = np.stack(betas,axis=0)
        guess[self.slices['betas']] = np.reshape(betas, self.nbetas)
        
        if self.ny > 1:
            guess[self.slices['sigmas']] = np.stack([np.diag(cov) for cov in covs], axis=0).reshape(self.nsigmas)
        else:
            guess[self.slices['sigmas']] = np.stack(covs, axis=0).reshape(self.nsigmas)
        
        # lower bound
        lb = np.array([-np.inf]*len(guess))
        lb[self.slices['sigmas']] = 1e-10
        bnds = Bounds(lb)

        print('EM Estimation Started...') if disp else None        
        thetas = guess
        start = time.time()
        gap = np.inf
        if (not get_ipython()) and plot:
            pyplot.ion()
            fig, ax = pyplot.subplots(1, 1, figsize = (14,8)) 
        for stepi in range(maxiter):
            postiors = self.postior(self.X, self.y, thetas)
            res = minimize(lambda thetas: -self.logLik(inputX, inputy, thetas, postiors) + self.alpha * self.penalty(thetas), 
                           thetas, method = 'SLSQP', bounds=bnds, options={'disp': False})
            gap = np.sum(np.abs(res.x - thetas))
            thetas = res.x
            print('Step {}: Log-likeihood = {:.4f}, gap = {:.4f}'.format(stepi, -res.fun, gap)) if disp else None
            if plot:
                if not get_ipython(): pyplot.cla()
                plotx = self.Xcol[0] if plotx is None else plotx
                ploty = self.ycol[0] if ploty is None else ploty
                if get_ipython(): 
                    self.plot(thetas, x=plotx, y=ploty)
                else:
                    self.plot(thetas, x=plotx, y=ploty, ax = ax, show=False)
                clear_output(wait = True)
                pyplot.pause(0.1)
            flag = res.success
            if gap < tol:
                break
            if stepi == maxiter-1:
                print('Warning: maximum number of iteration reached.')
                flag = False
        if not get_ipython():
            pyplot.ioff()
            pyplot.show()
            pyplot.close()
        loglikval = self.logLik(inputX, inputy, thetas, postiors)
        end = time.time()
        print('[EM Estimation]\t Completed in {:.4f} seconds.'.format(end-start)) if disp else None
        return thetas, loglikval, flag
    
    def bootstrapFit(self, nboot = 100, maxiter = 100, tol = 1e-4, disp = True):
        """
        Perform bootstrapping to estimate parameters.

        Args:
        - nboot: Number of bootstrapping iterations.
        - maxiter: Maximum number of iterations for each bootstrapped EM estimation.
        - tol: Tolerance for convergence.
        - disp: Boolean indicating whether to display progress.

        Returns:
        - Numpy array of bootstrapped parameter estimates.
        """
        boot_thetas = []
        print('Bootstrapping Model Fit Started...') if disp else None        
        with Timer('Bootstrapping', display=disp):
            for booti in tqdm.trange(nboot, disable = not disp, desc = 'Bootstrapping', leave = False):
                bootstrap_data = resample(self.X, self.y, n_samples = self.nobs, replace = True)
                Xs, ys = bootstrap_data[0], bootstrap_data[1]
                thetas, loglikval, flag = self.modelFit(Xs, ys, maxiter = maxiter, tol = tol, disp = False, plot = False)
                boot_thetas.append(thetas)
        return np.vstack(boot_thetas)
    
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
        loglik_gradient = gradient(thetas, lambda thetas: np.sum(probs * np.log(self.partialLik(X, y, thetas)), axis=1) )
        hex = np.zeros((len(thetas), len(thetas)))
        for i in range(self.nobs):
            hex += loglik_gradient[:,i][:, np.newaxis].dot(loglik_gradient[:,i][:, np.newaxis].T)
        return hex/self.nobs

    def targetBayes(self, phis:np.array, prior: np.array, postior: np.array):
        phis = np.reshape(phis, (self.ngroup-1, self.ngroup))
        probs = np.hstack([np.exp(phis.dot(prior.T)).T, np.ones((prior.shape[0], 1))])
        probs = probs / np.repeat(np.sum(probs, axis=1)[:, np.newaxis], self.ngroup, axis=1)
        return np.sum((probs - postior)**2)
    
    def solveBayes(self, prior: np.array, postior: np.array):
        if self.ngroup == 1:
            return prior
        phis = np.zeros((self.ngroup-1, self.ngroup))
        res = minimize(lambda phis: self.targetBayes(phis, prior, postior), phis[:], options={'disp': False})
        phis = np.reshape(res.x, (self.ngroup-1, self.ngroup))
        return phis

    def bayes(self, phis:np.array, prior: np.array):
        if self.ngroup == 1:
            return prior
        phis = np.reshape(phis, (self.ngroup-1, self.ngroup))
        probs = np.hstack([np.exp(phis.dot(prior.T)).T, np.ones((prior.shape[0], 1))])
        probs = probs / np.repeat(np.sum(probs, axis=1)[:, np.newaxis], self.ngroup, axis=1)
        return probs
    
    def fit(self, maxiter = 100, tol = 1e-4, boot = False, nboot = 100, disp = True, plot = True, plotx = None, ploty = None):
        """
        Main fitting function.

        Args:
        - maxiter: Maximum number of iterations for the EM algorithm.
        - tol: Tolerance for convergence.
        - boot: Boolean indicating whether to perform bootstrapping.
        - nboot: Number of bootstrapping iterations.
        - disp: Boolean indicating whether to display progress.
        - plot: Boolean indicating whether to plot the model during training.
        - plotx: Name of the x-axis variable for plotting.
        - ploty: Name of the y-axis variable for plotting.

        Returns:
        - Numpy array of final estimated parameters.
        """
        self.boot = boot
        if boot:
            self.boot_thetas = self.bootstrapFit(nboot = nboot, maxiter = maxiter, tol = tol, disp = disp)
        thetas, loglikval, flag = self.modelFit(self.X, self.y, maxiter = maxiter, tol = tol, 
                                          disp = disp, plot = plot, plotx = plotx, ploty = ploty)
        self.thetas = thetas
        self.loglikval = loglikval
        self.flag = flag
        priors = self.prior(self.X, thetas)
        postiors = self.postior(self.X, self.y, thetas)
        self.phis = self.solveBayes(priors, postiors)
        
        hex = self.hessian(self.X, self.y, thetas, postiors)
        self.varthetas = np.linalg.inv(hex)/(self.nobs - len(thetas))
        self.bicval = self.bic(loglikval, thetas, postiors)
        return thetas
    
    def deltaMethod(self, thetas: np.array, varthetas: np.array):
        """
        Computes the variance covariance matrix using delta method.

        Input:
        - thetas: Numpy array containing the parameters of the model.
        - varthetas: 2-D Numpy array containing the variance covariance matrix of the model parameters.

        Output:
        - BIC value.
        """
        delta = gradient(thetas, lambda thetas: self.__unpack(thetas)[-1])
        if self.cov:
            delta = np.concatenate([delta[self.slices['sigmas']], delta[self.slices['covs']]], axis = 0)  
        else:
            delta = delta[self.slices['sigmas']]
        
        varcov_vecs = []
        for g in range(self.ngroup):
            deltag = delta[:, g, :, :].reshape(delta.shape[0], self.ny*self.ny)
            varmtx = varthetas[-delta.shape[0]:, -delta.shape[0]:]
            varcov_vec = np.diag(deltag.T.dot(varmtx).dot(deltag)).reshape(self.ny, self.ny)
            varcov_vecs.append(varcov_vec)
        return np.array(varcov_vecs)

    def bic(self, loglik: float, thetas: np.array, postiors: np.array):
        """
        Computes the Bayesian Information Criterion (BIC) for the model.

        Input:
        - thetas: Numpy array containing the parameters of the model.

        Output:
        - BIC value.
        """
        Ktheta = len(thetas)
        logN = np.log(self.nobs)
        bicval = Ktheta*logN - 2*loglik
        return bicval
    
    def __predFunc(self, xi: int, X: np.array, thetas: np.array, mode: Literal['prior', 'postior']):
        guess = np.zeros(shape = (1, self.ny))
        priors = self.prior(X[xi,:][np.newaxis,:], thetas)
        if mode == 'prior':
            res = minimize(lambda y: -self.logLik(X[xi,:][np.newaxis,:], y, thetas, priors), 
                        guess, method = 'SLSQP', options={'disp': False})
            return res.x
        if mode == 'postior':
            postiors = self.bayes(self.phis, priors)
            groupid = np.argmax(postiors, axis=1)[0]
            gammas, betas, sigmas = self.__unpack(thetas)
            return X[xi,:].dot(betas[groupid]) 
    
    def predictDistr(self, X: pd.DataFrame = None):
        X = self.X if X is None else X
        if type(X) is pd.DataFrame:
            X = np.hstack((X[self.Xcol].values, np.ones(shape=(X.shape[0], 1)))) if self.const else X[self.Xcol].values
        priors = self.prior(X, self.thetas)
        gammas, betas, sigmas = self.__unpack(self.thetas)        
        return X.dot(betas.T), priors, sigmas

    def predict(self, X: pd.DataFrame = None, disp = True, lb = 0.05, ub = 0.95, mode: Literal['prior', 'postior'] = 'prior'):
        """
        Generates predictions using the trained model.

        Input:
        - X: Pandas dataframe of independent variables (optional, default is None, which uses the training data).
        - disp: display options
        - mode: 'prior' or 'postior' prediction method. Default is prior.

        Output:
        - Numpy array of predicted values for the dependent variable.
        """
        X = self.X if X is None else X
        if type(X) is pd.DataFrame:
            X = np.hstack((X[self.Xcol].values, np.ones(shape=(X.shape[0], 1)))) if self.const else X[self.Xcol].values
        y = []
        stds = []
        print('Model Prediction Started...')

        with Timer('Model Prediction'):
            for xi in tqdm.trange(X.shape[0], disable = not disp, desc = 'Model Prediction', leave = False):
                yi = self.__predFunc(xi, X, self.thetas, mode)
                gradyi = gradient(self.thetas, lambda thetas: self.__predFunc(xi, X, thetas, mode))
                predvars = gradyi.T.dot(self.varthetas).dot(gradyi)
                y.append(yi)
                stds.append(np.sqrt(np.diag(predvars)))
            preds = np.array(y)
            stds = np.array(stds)
        
        if not self.boot:
            return preds, stds
        print('Bootstrapping Prediction Started...') if disp else None
        start = time.time()
        bootpreds = []
        for booti in tqdm.trange(self.boot_thetas.shape[0], disable = not disp, desc = 'Bootstrapping', leave = False):
            booty = []
            for xi in range(X.shape[0]):
                guess = np.zeros(shape = (1, self.ny))
                priors = self.prior(X[xi,:][np.newaxis,:], self.boot_thetas[booti,:])
                res = minimize(lambda y: -self.logLik(X[xi,:][np.newaxis,:], y, self.boot_thetas[booti,:], priors), 
                            guess, method = 'SLSQP', options={'disp': False})
                yi = res.x
                booty.append(yi)
            bootpreds.append(np.array(booty).reshape((1, np.array(booty).size)))
        end = time.time()
        print('Bootstrap Completed in {:.4f} seconds.'.format(end-start)) if disp else None
        bootpreds = np.vstack(bootpreds)
        stds = np.std(bootpreds, axis=0).reshape(preds.shape)
        lbs = np.percentile(bootpreds, q = lb, axis=0).reshape(preds.shape)
        ubs = np.percentile(bootpreds, q = ub, axis=0).reshape(preds.shape)
        return preds, stds, lbs, ubs

    def plot(self, thetas, x, y, ax = None, figsize = (14,8), truelabel = None, show = True):
        """
        Plots the data points, model predictions, and uncertainty intervals.

        Input:
        - x: Name of the x-axis variable for plotting.
        - y: Name of the y-axis variable for plotting.
        - figsize: Tuple specifying the size of the plot.
        - truelabel: label of the true data points if available.

        Output: None
        """
        gammas, betas, sigmas = self.__unpack(thetas)
        if ax is None:
            fig, ax = pyplot.subplots(1, 1, figsize = figsize) 
        ls = []
        for g in range(self.ngroup):
            ax.scatter(self.data[x][np.argmax(self.postior(self.X, self.y, thetas), axis=1) == g], 
                   self.data[y][np.argmax(self.postior(self.X, self.y, thetas), axis=1) == g], 
                   color = colormap[g], label = 'state '+str(g)+ ' data')
            xaxis = np.linspace(self.data[x].min() - 0.05*(self.data[x].max() - self.data[x].min()), 
                                self.data[x].max() + 0.05*(self.data[x].max() - self.data[x].min()), 10)
            otherxs = copy.deepcopy(self.Xcol)
            otherxs.remove(x)
            beta = pd.DataFrame(betas[g,:,:].T, index = self.ycol, columns = self.Xcol + ['Const'])
            intercept = beta.loc[y, 'Const']
            for otherx in otherxs:
                intercept += self.data[otherx].mean()*beta.loc[y, otherx]
            ls.append(ax.plot(xaxis, intercept + beta.loc[y, x]*xaxis, label = 'state '+str(g)+ ' model', color = colormap[g]))
            sigma = pd.DataFrame(sigmas[g,:,:], index = self.ycol, columns = self.ycol)
            ax.fill_between(xaxis, intercept + beta.loc[y, x]*xaxis + 1.96 * np.sqrt(sigma.loc[y,y]),
                            intercept + beta.loc[y, x]*xaxis - 1.96 * np.sqrt(sigma.loc[y,y]), alpha = 0.2, color = colormap[g])
            ax.fill_between(xaxis, intercept + beta.loc[y, x]*xaxis + 1.68 * np.sqrt(sigma.loc[y,y]),
                            intercept + beta.loc[y, x]*xaxis - 1.68 * np.sqrt(sigma.loc[y,y]), alpha = 0.4, color = colormap[g])
        if truelabel in self.data.columns: ax.scatter(self.data[x], self.data[y], c = [colormap[c+2] for c in self.data[truelabel]], marker='x', s = 10)
        ax.legend(frameon = False, fontsize = 12, loc = 'upper left', ncol = self.ngroup)
        [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title('Scatter ' + x + '-' + y, fontsize=14)
        if show: pyplot.show()
        return None

    def plotMSE(self, y, figsize = (14,8), showci = True, alpha = 0.1, show = True, mode: Literal['prior', 'postior'] = 'prior'):
        """
        Plots the true data points against predicted data points with Mean Squared Error information.

        Input:
        - y: Name of the dependent variable for plotting.
        - figsize: Tuple specifying the size of the plot.

        Output: None
        """
        if self.boot:
            ypred, stds, lbs, ubs = self.predict()
            stds = pd.DataFrame(stds, columns=self.ycol)
            lbs = pd.DataFrame(lbs, columns=self.ycol)
            ubs = pd.DataFrame(ubs, columns=self.ycol)
        else:
            ypred, stds = self.predict(mode = mode)
        
        ypred = pd.DataFrame(ypred, columns=self.ycol)
        stds = pd.DataFrame(stds, columns=self.ycol)
        lbs = pd.DataFrame(ypred - norm.ppf(1.0-alpha/2.0)*stds, columns=self.ycol)
        ubs = pd.DataFrame(ypred + norm.ppf(1.0-alpha/2.0)*stds, columns=self.ycol)
        fig, ax = pyplot.subplots(1, 1, figsize = figsize)
        line45 = [min(self.data[y].min(), ypred[y].min()), max(self.data[y].max(), ypred[y].max())]
        ax.plot(line45, line45, '--', color = 'black')
        for g in range(self.ngroup):
            if self.boot:
                lbg = lbs[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g]
                ubg = ubs[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g]
                avg = (lbg.values + ubg.values)/2
                ax.scatter(self.data[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g],
                           ypred[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g], 
                           s = 5, alpha = 0.5, color = colormap[g], label = 'state '+str(g))
                if showci:
                    ax.errorbar(x = self.data[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g], y = avg, 
                                yerr = np.vstack([avg.reshape((1, avg.size)) - lbg.values.reshape((1, lbg.size)), 
                                                ubg.values.reshape((1, ubg.size)) - avg.reshape((1, avg.size))]),
                                marker = None, color = colormap[g], label = 'state ' + str(g) + 'Bootstrapped CI',
                                elinewidth=2, capsize=3, linewidth = 0)
            else:
                lbg = lbs[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g]
                ubg = ubs[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g]
                avg = (lbg.values + ubg.values)/2
                if showci:
                    ax.errorbar(x = self.data[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g],
                                y = ypred[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g],
                                yerr = np.vstack([avg.reshape((1, avg.size)) - lbg.values.reshape((1, lbg.size)), 
                                                ubg.values.reshape((1, ubg.size)) - avg.reshape((1, avg.size))]),
                                marker = 'o', color = colormap[g], label = 'state '+str(g)+ ' {:d}'.format(int(100*(1-alpha/2.0)))+ '%CI',
                                elinewidth=2, capsize=3, linewidth = 0)
                else:
                    ax.scatter(self.data[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g],
                               ypred[y][np.argmax(self.prior(self.X, thetas=self.thetas), axis = 1) == g], 
                               color = colormap[g], label = 'state '+str(g))
                
        [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(frameon = False, fontsize = 12, loc = 'upper left')
        ax.set_xlabel('True Data', fontsize=12)
        ax.set_ylabel('Pred Data', fontsize=12)
        ax.set_title('True Data vs Pred Data: ' + y + ', Mean Squared Error: {:.4f}'.format(mean_squared_error(self.data[y], ypred[y])), fontsize=14)
        if show: pyplot.show()
        return None if show else fig, ax
    
    def summary(self):
        """
        Prints a summary of the model parameters, log-likelihood, and BIC.
        """
        gammas, betas, sigmas = self.__unpack(self.thetas)
        gammastds, betastds, _ = self.__unpack(np.sqrt(np.diag(self.varthetas)))
        varcov_vecs = self.deltaMethod(self.thetas, self.varthetas)
        sigmastds = np.sqrt(varcov_vecs)
        print('#'+'-'*91+'#')
        print('{:^10s}'.format('Model Parameters').center(93))
        print('#'+'-'*91+'#')
        print(' '*5+'Observation:'.ljust(20), '{}'.format(self.nobs).ljust(30), 'Success Flag:'.ljust(20), '{}'.format(self.flag).ljust(20))
        print(' '*5+'Alpha:'.ljust(20), '{}'.format(self.alpha).ljust(30), 'LP Norm:'.ljust(20), '{}'.format(self.norm).ljust(20))
        print(' '*5+'Param Num:'.ljust(20), '{}'.format(len(self.thetas)).ljust(30), 'DF:'.ljust(20), '{}'.format(self.nobs - len(self.thetas)).ljust(20))
        print(' '*5+'Log-lik:'.ljust(20), '{:.4f}'.format(self.loglikval).ljust(30), 'BIC:'.ljust(20), '{:.4f}'.format(self.bicval).ljust(20))
        print('\n')
        print('Gamma: Logit Regression Coefficients')
        gammas = pd.DataFrame(gammas, index = ['state '+str(x) for x in range(self.ngroup-1)], columns = self.Xcol + ['Const'])
        gammastds = pd.DataFrame(gammastds, index = ['state '+str(x) for x in range(self.ngroup-1)], columns = self.Xcol + ['Const'])
        gammapvals = pd.DataFrame(norm.cdf(-np.abs(gammas.values/gammastds.values))*2, 
                                  index = ['state '+str(x) for x in range(self.ngroup-1)], columns = self.Xcol + ['Const'])

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
                      '{:.4f}'.format(gammastds.loc[index, col]).center(20), '|',
                      '{:.4f}'.format(gammapvals.loc[index, col]).center(20), '|',
                      )
        print('='*93)
        print('\n')
        print('Beta: Main Model Regression Coefficients')
        for g in range(self.ngroup):
            print('='*93)
            print('{:^10s}'.format('State ' + str(g)).center(93))
            betag = pd.DataFrame(betas[g,:,:].T, index = self.ycol, columns = self.Xcol + ['Const'])
            betastdg = pd.DataFrame(betastds[g,:,:].T, index = self.ycol, columns = self.Xcol + ['Const'])
            betapvalg = pd.DataFrame(norm.cdf(-np.abs(betag.values/betastdg.values))*2, 
                                  index = self.ycol, columns = self.Xcol + ['Const'])

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
                        '{:.4f}'.format(betastdg.loc[index, col]).center(20), '|',
                        '{:.4f}'.format(betapvalg.loc[index, col]).center(20), '|',
                        )
            print('='*93)
            print('\n')
        
        print('Sigma: Estimation Error Variance-Covariance Matrix')
        for g in range(self.ngroup):
            print('='*93)
            print('{:^10s}'.format('State ' + str(g)).center(93))
            sigmag = pd.DataFrame(sigmas[g,:,:], index = self.ycol, columns = self.ycol)
            sigmastdg = pd.DataFrame(sigmastds[g,:,:], index = self.ycol, columns = self.ycol)
            sigmapvalg = pd.DataFrame(norm.cdf(-np.abs(sigmag.values/sigmastdg.values))*2, index = self.ycol, columns = self.ycol)
            print('-'*93)
            print('|', 'vars-vars'.center(20), '|',  
                  'sigma'.center(20), '|',  
                  'std err'.center(20), '|',  
                  'p value'.center(20), '|')
            for idyi in range(self.ny):
                for idyj in range(idyi+1):
                    print('-'*93)
                    sigmaid = self.ycol[idyi][:8]+'-'+self.ycol[idyj][:8]
                    print('|', '{:^10s}'.format(sigmaid).center(20), '|', 
                          '{:.4f}'.format(sigmag.iloc[idyi, idyj]).center(20), '|',  
                          '{:.4f}'.format(sigmastdg.iloc[idyi, idyj]).center(20), '|',
                          '{:.4f}'.format(sigmapvalg.iloc[idyi, idyj]).center(20), '|',)
            print('='*93)
            print('\n')

        print('#'+'-'*91+'#')

if __name__ == "__main__":
    # test package
    from data_generator import PanelGenerator
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    mses = dict()
    data = PanelGenerator(ny = 2, ngroup=2, Xrange=(-3,3), seed = 658)
    data.summary()
    train, test = train_test_split(data.data, test_size = 0.2)
    
    # baseline regression
    lr = LinearRegression().fit(X = train[data.Xcol], y = train[data.ycol])
    predslr = lr.predict(test[data.Xcol])
    mses['lr'] = mean_squared_error(test[data.ycol], predslr, multioutput = 'raw_values')

    # GMLR regression
    gmlr = GMLR(train, ycol=data.ycol, Xcol=data.Xcol, alpha=0, ngroup=2, cov=True)
    thetas = gmlr.fit(maxiter=200, disp=True, plot=True, boot = False)
    gmlr.summary()
    gmlr.plot(thetas, x = data.Xcol[0], y = data.ycol[0], truelabel = 'group', show = True)

    '''
    # Prediction
    gmlr.plotMSE(y = 'y1')
    predsglmr, _ = gmlr.predict(test)
    gmlr.predictDistr()
    mses['gmlr'] = mean_squared_error(test[data.ycol], predsglmr, multioutput = 'raw_values')
    print(mses)
    '''
    
    means, priors, sigmas = gmlr.predictDistr()
    cd = CombinedDistr(priors[50,:], means[50,:], sigmas)
    print(gmlr.y[50,:])
    cd.plot(margin = 0)