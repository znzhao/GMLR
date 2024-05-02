import time
import copy
import warnings
import tqdm
import pandas as pd
import numpy as np
from numba import njit
from numba.np.unsafe import ndarray # essential to solve the bug in numba
from matplotlib import pyplot
from scipy.stats import norm
from scipy.optimize import Bounds
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from IPython.display import clear_output
from IPython import get_ipython
from helper.utils import Timer, gradient, dot3d
from data_generator import TSGenerator
warnings.filterwarnings('ignore')


@njit
def unpackParams(thetas: np.array, cov: bool, const:bool, ngroup: int, nX: int, ny: int, ngammas: int, netas:int, nbetas:int, nsigmas:int, ncovs:int):
    """
    Unpacks the flattened parameter array into individual matrices for further use.
    Parameters:
    - thetas: Numpy array containing the parameters of the model.

    Returns:
    - Tuple of gammas, betas, and sigmas (parameters of the model).
    """
    if cov:
        assert len(thetas) == ngammas + netas + nbetas + nsigmas + ncovs
    else:
        assert len(thetas) == ngammas + netas + nbetas + nsigmas

    gammas = thetas[ : ngammas]
    gammas = np.reshape(gammas, (ngroup-1, nX))
    etas = thetas[ngammas : ngammas+netas]
    etas = np.reshape(etas, (ngroup-1, ngroup-1 if const else ngroup))
    betas = thetas[ngammas+netas : ngammas+netas+nbetas]
    betas = np.reshape(betas, (ngroup, nX, ny))
    sigmas = thetas[ngammas + netas + nbetas : ngammas + netas + nbetas + nsigmas]
    sigmas = np.reshape(sigmas, (ngroup, ny))
    sigmalist = [np.array(np.diag(sigmas[g])) for g in range(ngroup)]
    sigmas = np.zeros((ngroup, ny, ny))
    for g in range(ngroup):
        sigmas[g,:,:] = sigmalist[g]

    if cov and int(ny*(ny-1)/2)>0:
        # lower trianglular decomposition factors
        covs = thetas[ngammas + netas + nbetas + nsigmas : ngammas + netas + nbetas + nsigmas + ncovs]
        covs = np.reshape(covs, (ngroup, int(ny*(ny-1)/2)))
        for g in range(ngroup):
            k = 0
            for i in range(1, ny):
                for j in range(0, i):
                    sigmas[g, i, j] = covs[g, k]
                    k += 1
    
    for g in range(ngroup):
        sigmas[g] = sigmas[g].dot(sigmas[g].T)
        
    return gammas, etas, betas, sigmas

@njit
def update(X: np.array, y: np.array, thetas: np.array, cov: bool, const:bool, nobs: int, ngroup: int, 
           nX: int, ny: int, ngammas: int, netas:int, nbetas:int, nsigmas:int, ncovs:int, init_guess = None):        
    # use Bayes Rule to calculate the prior and the postior simultaneously for each state
    if init_guess is None:
        init_guess = np.zeros(shape=(1, ngroup - 1 if const else ngroup))
    gammas, etas, betas, sigmas = unpackParams(thetas, cov, const, ngroup, nX, ny, ngammas, netas, nbetas, nsigmas, ncovs)
    y = y.reshape(X.shape[0], ny)
    ymtx = np.zeros((X.shape[0], ny, ngroup))
    for g in range(ngroup):
        ymtx[:,:,g] = y
    err = ymtx - dot3d(X, betas)

    lagP = init_guess
    priors = np.zeros((nobs, ngroup))
    postiors = np.zeros((nobs, ngroup))
    for t in range(nobs):
        denominator = 1 + np.sum(np.exp(X[t,:].dot(gammas.T) + lagP.dot(etas.T)))
        nominator = np.hstack((np.exp(X[t,:].dot(gammas.T) + lagP.dot(etas.T)), np.ones((1, 1))))
        prior = nominator / denominator
        priors[t, :] = prior
        pmtx = np.zeros((1, ngroup))
        for g in range(ngroup):
            # Probability that observation falls in the i-th group
            curr_err = err[t,:,g][np.newaxis,:]
            normalprob = np.exp(-1.0/2.0 * np.diag(curr_err.dot(np.linalg.inv(sigmas[g])).dot(curr_err.T)))
            pval = prior[0, g] / np.sqrt(2.0*np.pi*np.linalg.det(sigmas[g])) * normalprob # ??????
            pmtx[:, g] = pval
        postior = pmtx / np.sum(pmtx, axis=1)
        postiors[t, :] = postior
        lagP = postior[:, :-1] if const else postior
    return priors, postiors

class MSGMLR:
    colormap = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:blue', 
                5:'tab:purple', 6:'tab:brown', 7:'tab:pink', 8:'tab:gray', 9:'tab:olive', 10:'tab:cyan'}
    def __init__(self, data: pd.DataFrame, ycol: list, Xcol: list, ngroup = 2, const = True, cov = False, alpha = 0, norm = 1):
        """
        Initialize the Markov Switching Gaussian Mixture Linear Regression (GMLR) model with adjustable state-dependent probability.

        Parameters:
        - data: Pandas DataFrame containing the dataset. The index of the dataset should be the time index.
        - ycol: List of column names for the dependent variable(s).
        - Xcol: List of column names for the independent variable(s).
        - ngroup: Number of groups or states in the model.
        - const: Boolean indicating whether to include a constant term in the regression model.
        - cov: Boolean indicating whether to include a covariance term in the variance-covariance matrix.
        ---------------------------------------------------------------------------------------------
        The MSGMLR class is designed for fitting and predicting a mixture model for time series data using the Expectation-Maximization 
        (EM) algorithm. To initialize the class, create an instance by providing a dataset in the form of a Pandas DataFrame (`data`), 
        specifying the column names for the dependent variables (`ycol`), independent variables (`Xcol`), the number of groups or states
        (`ngroup`), and whether to include a constant term in the regression model (`const`). The instance is initialized with these 
        parameters. Once initialized, the `fit` method can be called to train the model using the EM algorithm. It takes optional 
        parameters such as `maxiter` (maximum number of iterations), `tol` (convergence tolerance), `disp` (display progress), 
        `plot` (plotting the model during training), `plotx` (x-axis variable for plotting), and `ploty` (y-axis variable for plotting).
        After fitting, the `predict` method can be used to generate predictions for the dependent variable. If no independent variables
        are provided, it uses the training data by default. Additionally, the `summary` method can be called to print a summary of the 
        model parameters, log-likelihood, and Bayesian Information Criterion (BIC). Below is an example of using this class:

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
        self.X = np.hstack((data[Xcol].values, np.ones(shape=(self.nobs, 1)))) if const else data[Xcol].values
        self.nX = self.X.shape[1]
        self.Xcol = Xcol
        self.ngroup = ngroup
        self.ngammas = (self.ngroup-1)*self.nX
        self.nbetas = self.nX*self.ngroup*self.ny
        self.nsigmas = self.ngroup*self.ny
        self.ncovs = self.ngroup*int(self.ny*(self.ny-1)/2)
        self.netas = (self.ngroup-1)**2 if const else self.ngroup*(self.ngroup-1)

        self.slices = dict()
        self.slices['gammas'] = np.index_exp[:self.ngammas]
        self.slices['etas']   = np.index_exp[self.ngammas: 
                                             self.ngammas + self.netas]
        self.slices['betas']  = np.index_exp[self.ngammas + self.netas:
                                             self.ngammas + self.netas + self.nbetas]
        self.slices['sigmas'] = np.index_exp[self.ngammas + self.netas + self.nbetas: 
                                             self.ngammas + self.netas + self.nbetas + self.nsigmas]
        self.slices['covs']   = np.index_exp[self.ngammas + self.netas + self.nbetas + self.nsigmas: 
                                             self.ngammas + self.netas + self.nbetas + self.nsigmas + self.ncovs]
        
        self.cov = cov
        self.alpha = alpha
        self.norm = norm
        self.const = const

    def __unpack(self, thetas: np.array):
        return unpackParams(thetas, self.cov, self.const, self.ngroup, self.nX, self.ny, self.ngammas, self.netas, self.nbetas, self.nsigmas, self.ncovs)

    def unpack(self, thetas: np.array):
        """
        Public interface to unpack the model parameters.

        Parameters:
        - thetas: Numpy array containing the parameters of the model.

        Returns:
        - Tuple of gammas, betas, and sigmas (parameters of the model).
        """
        return self.__unpack(thetas)

    def update(self, X: np.array, y: np.array, thetas: np.array, init_guess = None):
        nobs = X.shape[0]
        return update(X, y, thetas, self.cov, self.const, nobs, self.ngroup, self.nX, self.ny, self.ngammas, self.netas, self.nbetas, self.nsigmas, self.ncovs, init_guess)
    
    def partialLik(self, X: np.array, y: np.array, thetas: np.array, init_guess = None):
        """
        Computes the partial likelihood function for the model.

        Parameters:
        - X: Numpy array of independent variables.
        - y: Numpy array of dependent variables.
        - thetas: Numpy array containing the parameters of the model.
        - probs: Numpy array of the prior probability.

        Returns:
        - Value of the partial likelihood function.
        """
        gammas, etas, betas, sigmas = self.__unpack(thetas)
        priors, postiors = self.update(self.X, self.y, thetas, init_guess)
        y = y.reshape(X.shape[0], self.ny)
        ymtx = np.repeat(y[:, :, np.newaxis], self.ngroup, axis=2)
        err = ymtx - X.dot(betas.T)
        
        pmtx = []
        for g in range(self.ngroup):
            # Probability that observation falls in the i-th group
            normalprob = np.exp(-1.0/2.0 * np.diag(err[:,:,g].dot(np.linalg.inv(sigmas[g])).dot(err[:,:,g].T)))
            pval = priors[:,g] / np.sqrt(2.0*np.pi*np.linalg.det(sigmas[g])) * normalprob
            pmtx.append(pval)
        return np.array(pmtx).T
    
    def filter(self, X: np.array, y: np.array, thetas: np.array, priors: np.array, postiors: np.array):
        gammas, etas, betas, sigmas = self.__unpack(thetas)
        nobs = X.shape[0]
        probs = np.zeros(shape=(nobs, self.ngroup))
        curr_smoothed_prob = probs[nobs-1,:] = postiors[-1,:]
        for t in range(nobs-1):
            pt = nobs-t-1
            lagPs = np.zeros((self.ngroup, self.ngroup-1)) if self.const else np.zeros((self.ngroup, self.ngroup))
            for g in range(self.ngroup-1 if self.const else self.ngroup):
                lagPs[g, g] = 1
            wedge = np.zeros(self.ngroup)
            Ptrans = np.zeros((self.ngroup, self.ngroup))
            for g in range(self.ngroup):
                lagP = lagPs[g, :]
                denominator = 1 + np.sum(np.exp(X[pt,:].dot(gammas.T) + lagP.dot(etas.T)))
                nominator = np.hstack((np.exp(X[pt,:].dot(gammas.T) + lagP.dot(etas.T)), np.ones((1))))
                Ptrans[g,:] = nominator / denominator  # used in the nominator
            for g in range(self.ngroup):
                bottom = np.array([sum(Ptrans[:,x] * postiors[pt-1, :]) for x in range(self.ngroup)])
                wedge[g] = np.sum(Ptrans[g] * curr_smoothed_prob / bottom)
            curr_smoothed_prob = probs[pt-1] = wedge * postiors[pt-1, :]
        return probs
            
    
    def logLik(self, X: np.array, y: np.array, thetas: np.array, postiors: np.array, init_guess = None):
        """
        Computes the log-likelihood function.

        Input:
        - X: Numpy array of independent variables.
        - y: Numpy array of dependent variables.
        - thetas: Numpy array containing the parameters of the model.
        - probs: Numpy array containing postior probabilities.

        Output:
        - Log-likelihood value.
        """
        # take expectation of the log-likilihood function
        loglik = np.log(self.partialLik(X, y, thetas, init_guess))
        label = loglik == -np.inf
        loglik = np.sum(postiors * loglik)
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
            guess = np.zeros(self.ngammas + self.netas + self.nbetas + self.nsigmas + self.ncovs)
        else:
            guess = np.zeros(self.ngammas + self.netas + self.nbetas + self.nsigmas)

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
            fig, ax = pyplot.subplots(1, 1, figsize = (16,8)) 
        for stepi in range(maxiter):
            priors, postiors = self.update(self.X, self.y, thetas)
            smoothed = self.filter(self.X, self.y, thetas, priors, postiors)
            res = minimize(lambda thetas: -self.logLik(inputX, inputy, thetas, smoothed) + self.alpha * self.penalty(thetas), 
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
        raise NotImplementedError
    
    def hessian(self, X: np.array, y: np.array, thetas: np.array, postiors: np.array):
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
        loglik_gradient = gradient(thetas, lambda thetas: np.sum(postiors * np.log(self.partialLik(X, y, thetas)), axis=1) )
        hex = np.zeros((len(thetas), len(thetas)))
        for i in range(self.nobs):
            hex += loglik_gradient[:,i][:, np.newaxis].dot(loglik_gradient[:,i][:, np.newaxis].T)
        return hex/self.nobs

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
        priors, postiors = self.update(self.X, self.y, thetas)
        smoothed = self.filter(self.X, self.y, thetas, priors, postiors)
        hex = self.hessian(self.X, self.y, thetas, smoothed)
        self.varthetas = np.linalg.inv(hex)/(self.nobs - len(thetas))
        self.bicval = self.bic(loglikval, thetas, smoothed)
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
        delta = gradient(thetas, lambda thetas: self.__unpack(thetas)[-1])   ### ?
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
    
    def __predFunc(self, X: np.array, smoothed: np.array, thetas: np.array, init_guess = None):
        guess = np.zeros(shape = (1, self.ny))
        res = minimize(lambda y: -self.logLik(X, y, thetas, smoothed, init_guess), 
                       guess, method = 'SLSQP', options={'disp': False})
        return res.x
    
    def __predFuncOOS(self, X: np.array, postiors: np.array, thetas: np.array, init_guess = None):
        guess = np.zeros(shape = (1, self.ny))
        res = minimize(lambda y: -self.logLik(X, y, thetas, postiors, init_guess), 
                       guess, method = 'SLSQP', options={'disp': False})
        return res.x

    def predict(self, X: pd.DataFrame = None, disp = True):
        """
        Generates predictions using the trained model.

        Input:
        - X: Pandas Dataframe of independent variables (optional, default is None, which uses the training data).

        Output:
        - Numpy array of predicted values for the dependent variable.
        """
        if self.boot:
            raise NotImplementedError
        
        y = []
        stds = []
        if X is None:
            X = self.X
            print('Model In-Sample Prediction Started...')
            with Timer('Model Prediction'):
                priors, postiors = self.update(self.X, self.y, self.thetas)
                smoothed = self.filter(self.X, self.y, self.thetas, priors, postiors)
                for xi in tqdm.trange(X.shape[0], disable = not disp, desc = 'Model Prediction', leave = False):
                    init_guess = (smoothed[xi-1,:][np.newaxis,:-1] if self.const else smoothed[xi-1,:][np.newaxis,:]) if xi>0 else None
                    yi = self.__predFunc(X[xi,:][np.newaxis,:], priors[xi,:][np.newaxis,:], self.thetas, init_guess)                    
                    gradyi = gradient(self.thetas, lambda thetas: self.__predFunc(X[xi,:][np.newaxis,:], priors[xi,:][np.newaxis,:], thetas, init_guess))
                    predvars = gradyi.T.dot(self.varthetas).dot(gradyi)
                    y.append(yi)
                    stds.append(np.sqrt(np.diag(predvars)))
                preds = np.array(y)
                stds = np.array(stds)
        else:
            if type(X) is pd.DataFrame:
                X = np.hstack((X[self.Xcol].values, np.ones(shape=(X.shape[0], 1)))) if self.const else X[self.Xcol].values
            print('Model Out-Of-Sample Prediction Started...')
            with Timer('Model Prediction'):
                priors, postiors = self.update(self.X, self.y, self.thetas)
                smoothed = self.filter(self.X, self.y, self.thetas, priors, postiors)
                curr_prob = smoothed[-1,:][np.newaxis,:-1] if self.const else smoothed[xi-1,:][np.newaxis,:]
                for xi in tqdm.trange(X.shape[0], disable = not disp, desc = 'Model Prediction', leave = False):
                    init_guess = curr_prob
                    # calculate the probability used in calculating the out-of-sample prediction
                    gammas, etas, betas, sigmas = self.__unpack(self.thetas)
                    denominator = 1 + np.sum(np.exp(X[xi,:][np.newaxis,:].dot(gammas.T) + init_guess.dot(etas.T)))
                    nominator = np.hstack((np.exp(X[xi,:][np.newaxis,:].dot(gammas.T) + init_guess.dot(etas.T)), np.ones((1, 1))))
                    prior = nominator / denominator
                    # do out-of-sample prediction
                    yi = self.__predFuncOOS(X[xi,:][np.newaxis,:], prior, self.thetas, init_guess)          
                    gradyi = gradient(self.thetas, lambda thetas: self.__predFuncOOS(X[xi,:][np.newaxis,:], prior, self.thetas, init_guess))
                    predvars = gradyi.T.dot(self.varthetas).dot(gradyi)
                    y.append(yi)
                    stds.append(np.sqrt(np.diag(predvars)))
                    priors, postiors = self.update(X[xi,:][np.newaxis,:], yi[np.newaxis,:], self.thetas, init_guess)
                    smoothed = self.filter(X[xi,:][np.newaxis,:], yi[np.newaxis,:], self.thetas, priors, postiors)
                    curr_prob = smoothed[-1,:][np.newaxis,:-1] if self.const else smoothed[xi-1,:][np.newaxis,:]
                    
                preds = np.array(y)
                stds = np.array(stds)
        return preds, stds

    def plot(self, thetas, x, y, ax = None, figsize = (16,8), truelabel = None, show = True):
        """
        Plots the data points, model predictions, and uncertainty intervals.

        Input:
        - x: Name of the x-axis variable for plotting.
        - y: Name of the y-axis variable for plotting.
        - figsize: Tuple specifying the size of the plot.
        - truelabel: label of the true data points if available.

        Output: None
        """
        gammas, etas, betas, sigmas = self.__unpack(thetas)
        if ax is None:
            fig, ax = pyplot.subplots(1, 1, figsize = figsize) 
        if truelabel in self.data.columns: ax.scatter(self.data[x], self.data[y], c = [self.colormap[c+2] for c in self.data['z']], marker='x')
        ls = []
        priors, postiors = self.update(self.X, self.y, thetas)
        smoothed = self.filter(self.X, self.y, thetas, priors, postiors)
        for g in range(self.ngroup):
            ax.scatter(self.data[x][np.argmax(smoothed, axis=1) == g], 
                   self.data[y][np.argmax(smoothed, axis=1) == g], 
                   color = self.colormap[g], label = 'state '+str(g)+ ' data')
            xaxis = np.linspace(self.data[x].min() - 0.05*(self.data[x].max() - self.data[x].min()), 
                                self.data[x].max() + 0.05*(self.data[x].max() - self.data[x].min()), 10)
            otherxs = copy.deepcopy(self.Xcol)
            otherxs.remove(x)
            beta = pd.DataFrame(betas[g,:,:].T, index = self.ycol, columns = self.Xcol + ['Const'])
            intercept = beta.loc[y, 'Const']
            for otherx in otherxs:
                intercept += self.data[otherx].mean()*beta.loc[y, otherx]
            ls.append(ax.plot(xaxis, intercept + beta.loc[y, x]*xaxis, label = 'state '+str(g)+ ' model', color = self.colormap[g]))
            sigma = pd.DataFrame(sigmas[g,:,:], index = self.ycol, columns = self.ycol)
            ax.fill_between(xaxis, intercept + beta.loc[y, x]*xaxis + 1.96 * np.sqrt(sigma.loc[y,y]),
                            intercept + beta.loc[y, x]*xaxis - 1.96 * np.sqrt(sigma.loc[y,y]), alpha = 0.2, color = self.colormap[g])
            ax.fill_between(xaxis, intercept + beta.loc[y, x]*xaxis + 1.68 * np.sqrt(sigma.loc[y,y]),
                            intercept + beta.loc[y, x]*xaxis - 1.68 * np.sqrt(sigma.loc[y,y]), alpha = 0.4, color = self.colormap[g])
        ax.legend(frameon = False, fontsize = 12, loc = 'upper left', ncol = self.ngroup)
        [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title('Scatter ' + x + '-' + y, fontsize=14)
        if show: pyplot.show()
        return None if show else ax

    def plotMSE(self, y, figsize = (16,8), showci = True, alpha = 0.1, show = True):
        """
        Plots the true data points against predicted data points with Mean Squared Error information.

        Input:
        - y: Name of the dependent variable for plotting.
        - figsize: Tuple specifying the size of the plot.

        Output: None
        """
        if self.boot:
            raise NotImplementedError
        else:
            ypred, stds = self.predict()
        
        ypred = pd.DataFrame(ypred, columns=self.ycol)
        stds = pd.DataFrame(stds, columns=self.ycol)
        lbs = pd.DataFrame(ypred - norm.ppf(1.0-alpha/2.0)*stds, columns=self.ycol)
        ubs = pd.DataFrame(ypred + norm.ppf(1.0-alpha/2.0)*stds, columns=self.ycol)
        fig, ax = pyplot.subplots(1, 1, figsize = figsize)
        line45 = [min(self.data[y].min(), ypred[y].min()), max(self.data[y].max(), ypred[y].max())]
        ax.plot(line45, line45, '--', color = 'black')
        for g in range(self.ngroup):
            if self.boot:
                raise NotImplementedError
            else:
                priors, postiors = self.update(self.X, self.y, thetas)
                smoothed = self.filter(self.X, self.y, thetas, priors, postiors)
                lbg = lbs[y][np.argmax(smoothed, axis = 1) == g]
                ubg = ubs[y][np.argmax(smoothed, axis = 1) == g]
                avg = (lbg.values + ubg.values)/2
                if showci:
                    ax.errorbar(x = self.data[y][np.argmax(smoothed, axis = 1) == g],
                                y = ypred[y][np.argmax(smoothed, axis = 1) == g],
                                yerr = np.vstack([avg.reshape((1, avg.size)) - lbg.values.reshape((1, lbg.size)), 
                                                ubg.values.reshape((1, ubg.size)) - avg.reshape((1, avg.size))]),
                                marker = 'o', color = self.colormap[g], label = 'state '+str(g)+ ' {:d}'.format(int(100*(1-alpha/2.0)))+ '%CI',
                                elinewidth=2, capsize=3, linewidth = 0)
                else:
                    ax.scatter(self.data[y][np.argmax(smoothed, axis = 1) == g],
                               ypred[y][np.argmax(smoothed, axis = 1) == g], 
                               color = self.colormap[g], label = 'state '+str(g))
                
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
        gammas, etas, betas, sigmas = self.__unpack(self.thetas)
        gammastds, etastds, betastds, _ = self.__unpack(np.sqrt(np.diag(self.varthetas)))
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
        print('Gamma & Eta: Logit Regression Coefficients')
        gammas = pd.DataFrame(gammas, index = ['state '+str(x) for x in range(self.ngroup-1)], columns = self.Xcol + ['Const'])
        gammastds = pd.DataFrame(gammastds, index = ['state '+str(x) for x in range(self.ngroup-1)], columns = self.Xcol + ['Const'])
        gammapvals = pd.DataFrame(norm.cdf(-np.abs(gammas.values/gammastds.values))*2, 
                                  index = ['state '+str(x) for x in range(self.ngroup-1)], columns = self.Xcol + ['Const'])
        eta_cols = ['P(L.state '+str(x)+')' for x in range(self.ngroup-1 if self.const else self.ngroup)]
        etas = pd.DataFrame(etas, index = ['state '+str(x) for x in range(self.ngroup-1)],
                            columns = eta_cols)
        etastds = pd.DataFrame(etastds, index = ['state '+str(x) for x in range(self.ngroup-1)], 
                               columns = eta_cols)
        etapvals = pd.DataFrame(norm.cdf(-np.abs(etas.values/etastds.values))*2, 
                                index = ['state '+str(x) for x in range(self.ngroup-1)], 
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
            for col in self.Xcol:
                print('|', '{:^10s}'.format(col).center(20), '|', 
                      '{:.4f}'.format(row[col]).center(20), '|',  
                      '{:.4f}'.format(gammastds.loc[index, col]).center(20), '|',
                      '{:.4f}'.format(gammapvals.loc[index, col]).center(20), '|',
                      )
            for col in eta_cols:
                print('|', '{:^10s}'.format(col).center(20), '|', 
                      '{:.4f}'.format(etas.loc[index, col]).center(20), '|',  
                      '{:.4f}'.format(etastds.loc[index, col]).center(20), '|',
                      '{:.4f}'.format(etapvals.loc[index, col]).center(20), '|',
                      )
            col = 'Const'
            print('|', '{:^10s}'.format(col).center(20), '|', 
                  '{:.4f}'.format(gammas.loc[index, col]).center(20), '|',  
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
                    print('|', '{:^10s}'.format(col).center(20), '|', 
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
                    print('|', '{:^10s}'.format(self.ycol[idyi]+'-'+self.ycol[idyj]).center(20), '|', 
                          '{:.4f}'.format(sigmag.iloc[idyi, idyj]).center(20), '|',  
                          '{:.4f}'.format(sigmastdg.iloc[idyi, idyj]).center(20), '|',
                          '{:.4f}'.format(sigmapvalg.iloc[idyi, idyj]).center(20), '|',)
            print('='*93)
            print('\n')

        print('#'+'-'*91+'#')

if __name__ == "__main__":
    # test package
    data = TSGenerator(nX=2, ny = 2, Xrange=(-3, 3))
    data.summary()
    msgmlr = MSGMLR(data.data, ycol=data.ycol, Xcol=data.Xcol, alpha=0, ngroup=2, cov=True)
    thetas = msgmlr.fit(maxiter=200, disp=True, plot=True, boot = False)
    priors, postiors = msgmlr.update(msgmlr.X, msgmlr.y, thetas)
    smoothed = msgmlr.filter(msgmlr.X, msgmlr.y, thetas, priors, postiors)
    label = smoothed[:,0] if np.mean((smoothed[:,0] - data.g)**2) < np.mean((smoothed[:,0] - 1 + data.g)**2) else 1-smoothed[:,0]

    # plot the state graph
    fig, ax = pyplot.subplots(1, 1, figsize = (16,8))
    ax.plot(data.data.index, data.g, label = 'True Probability')
    ax.plot(data.data.index, label, label = 'Smoothed Probability')
    [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(frameon = False, fontsize = 12, loc = 'upper left')
    ax.set_xlabel('True Data', fontsize=12)
    ax.set_ylabel('Pred Data', fontsize=12)
    ax.set_title('Smoothed Probability vs True Probability of being in state 1', fontsize=14)
    pyplot.show()
    
    # model results and prediction
    msgmlr.summary()
    msgmlr.plotMSE(y = 'y1')
    oosX = data.data[data.Xcol].iloc[:10,:].values
    oosX = np.hstack((oosX, np.ones(shape=(oosX.shape[0], 1))))
    predys, predsds = msgmlr.predict(X=oosX)
    print(predys)