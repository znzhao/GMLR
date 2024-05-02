import time
import copy
import warnings
from numpy.core.multiarray import array as array
import tqdm
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot
from IPython.display import clear_output
from IPython import get_ipython
from helper.utils import Timer, gradient
from data_generator import TSGenerator
from msgmlr import MSGMLR
warnings.filterwarnings('ignore')


class MSGMLVAR(MSGMLR):
    def __init__(self, data: pd.DataFrame, ycol: list = [], Xcol: list = [], extXcol: list = [], 
                 lags: list|int = 1, ngroup = 2, const = True, cov = False, alpha = 0, norm = 1):
        """
        Markov Switching Gaussian Mixture Model with Lagged Variables (MSGMLVAR) class.
        
        Args:
            data (pd.DataFrame): Input data.
            ycol (list): List of columns for the endogenous variables.
            Xcol (list): List of columns for exogenous variables.
            extXcol (list): List of columns for externally given exogenous variables.
            lags (list|int): Number of lags to consider.
            ngroup (int): Number of groups for the Gaussian mixture model.
            const (bool): Whether to include a constant term.
            cov (bool): Whether to calculate covariances.
            alpha (float): Coefficient for regularization.
            norm (float): Normalization factor.
        """
        self.origdata = data
        self.ycol = ycol if ycol != [] else data.columns.tolist()
        self.lagXcol = Xcol
        self.extXcol = extXcol
        self.nlags = len(lags) if type(lags) is list else lags
        self.lags = lags if type(lags) is list else range(1, lags+1)
        self.vardata, allXcol = self.initialize(data, ycol, Xcol, extXcol, lags)
        super().__init__(self.vardata, self.ycol, allXcol, ngroup, const, cov, alpha, norm)
    
    def initialize(self, data: pd.DataFrame):
        """
        Initialize the data for the VAR model.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            vardata (pd.DataFrame): DataFrame prepared for VAR model.
            allXcol (list): List of all columns used.
        """
        varX = [data[self.extXcol]] if self.extXcol != [] else []
        allXcol = []
        for lag in self.lags:
            lagvary = data[self.ycol].shift(lag)
            allXcol = allXcol + ['L{}.'.format(lag) + x for x in self.ycol]
            lagvary.columns = ['L{}.'.format(lag) + x for x in self.ycol]
            varX.append(lagvary)
            lagvarX = data[self.lagXcol].shift(lag)
            allXcol = allXcol + ['L{}.'.format(lag) + x for x in self.lagXcol]
            lagvarX.columns = ['L{}.'.format(lag) + x for x in self.lagXcol]
            varX.append(lagvarX)

        varX = pd.concat(varX, axis=1)
        vary = data[self.ycol]
        vardata = pd.concat([vary, varX], axis=1).dropna(axis=0)
        allXcol = allXcol + self.extXcol
        return vardata, allXcol

    def statePred(self, thetas: np.array, state = 0, periods = 5):
        """
        Predict state for a given number of periods.

        Args:
            thetas (np.array): Model parameters.
            state (int): State to predict.
            periods (int): Number of periods to predict.

        Returns:
            irf (np.array): Predicted state for each period.
        """
        gammas, etas, betas, sigmas = self.unpack(thetas)
        irf = np.zeros((periods, self.ny, self.ny))
        irf[0, :, :] = np.eye(self.ny)
        beta = betas[state, :, :]
        beta_lags = np.zeros((self.nlags, self.ny, self.ny))
        for lag in range(self.nlags):
            beta_lags[lag, :, :] = beta[self.ny*(lag):self.ny*(lag+1), :]
        for p in range(1, periods):
            for lag in range(min(self.nlags, p)):
                irf[p, :, :] += irf[p-lag-1, :, :].dot(beta_lags[lag, :, :])
        return irf
    
    def statePredStd(self, state = 0, periods = 5):
        """
        Calculate standard deviation of state prediction.

        Args:
            state (int): State to predict.
            periods (int): Number of periods.

        Returns:
            stds (np.array): Standard deviations of state prediction for each period.
        """
        grady = gradient(self.thetas, lambda thetas: self.statePred(thetas, state, periods))
        stds = np.zeros((periods, self.ny, self.ny))
        for p in range(periods):
            predvars = np.zeros((self.ny, self.ny))
            for shock_i in range(self.ny):
                for y_i in range(self.ny):
                    predvars[shock_i, y_i] = grady[:, p, shock_i, y_i].T.dot(self.varthetas).dot(grady[:, p, shock_i, y_i])
            stds[p, :, :] = np.sqrt(predvars)
        return stds
    
    def plotIrf(self, periods = 12, figsize = (16,8), showci = True, alpha = 0.1, show = True):
        """
        Plot impulse response functions (IRFs).

        Args:
            periods (int): Number of periods.
            figsize (tuple): Figure size.
            showci (bool): Whether to show confidence intervals.
            alpha (float): Confidence level.
            show (bool): Whether to display the plot.
        """
        if self.lagXcol != []: raise Exception("IRFs not avaible for models with external X.")
        
        fig, ax = pyplot.subplots(self.ny, self.ny, figsize = figsize)
        taxis = range(periods)
        for state in range(self.ngroup):
            irf = self.statePred(self.thetas, state, periods)
            if showci:
                stds = self.statePredStd(state, periods)
                lbs = irf - norm.ppf(1.0-alpha/2.0)*stds
                ubs = irf + norm.ppf(1.0-alpha/2.0)*stds
            for shock_i in range(self.ny):
                for y_i in range(self.ny):
                    ax[shock_i, y_i].plot(taxis, irf[:, shock_i, y_i], color = self.colormap[state], label = 'state '+str(state))
                    if showci:
                        ax[shock_i, y_i].fill_between(taxis, ubs[:, shock_i, y_i], lbs[:, shock_i, y_i], alpha = 0.4, color = self.colormap[state])

        for shock_i in range(self.ny):
            for y_i in range(self.ny):
                ax[shock_i, y_i].plot(taxis, np.zeros(periods), '--', color = 'black')
                fig.tight_layout()
                [ax[shock_i, y_i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
                ax[shock_i, y_i].tick_params(axis='both', which='major', labelsize=10)
                ax[shock_i, y_i].legend(frameon = False, fontsize = 12, loc = 'upper left')
                ax[shock_i, y_i].set_xlabel('Periods', fontsize=12)
                ax[shock_i, y_i].set_ylabel('Unit', fontsize=12)
                ax[shock_i, y_i].set_title(self.ycol[shock_i] + ' Shock - ' + self.ycol[y_i], fontsize=14)
                ax[shock_i, y_i].set_xticks(taxis)
        if show: pyplot.show()
        return None

    def plotProb(self, figsize = (16,8), show = True):
        fig, ax = pyplot.subplots(self.ngroup-1, figsize = figsize)
        priors, postiors = self.update(self.X, self.y, self.thetas)
        smoothed = self.filter(self.X, self.y, self.thetas, priors, postiors)
        for g in range(self.ngroup-1):
            curr_ax = ax if self.ngroup == 2 else ax[g]
            curr_ax.plot(self.data.index, smoothed[:, g])
            [curr_ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
            curr_ax.tick_params(axis='both', which='major', labelsize=10)
            curr_ax.set_xlabel('Date', fontsize=12)
            curr_ax.set_ylabel('Probability', fontsize=12)
            curr_ax.set_title('State '+ str(g) + ' Probability', fontsize=14)
        if show: pyplot.show()
        return None
    
    def predict(self, test: pd.DataFrame = None, disp=True):
        if test is None:
            return super().predict(test, disp)
        ntest = test.shape[0]
        vardata, allXcol = self.initialize(self, pd.concat([self.origdata, test], axis=1))
        vardata = vardata.iloc[-ntest:]
        return super().predict(vardata, disp)

if __name__ == "__main__":
    data = pd.read_excel('./data/Sample.xlsx', index_col=0)
    print(data.tail(200))
    msgmlvar = MSGMLVAR(data.tail(200), lags=4)
    thetas = msgmlvar.fit(plotx='L1.FFR')
    msgmlvar.summary()
    msgmlvar.plotIrf(showci = False)
    msgmlvar.plotProb()

