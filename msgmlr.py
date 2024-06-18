import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython import get_ipython
from IPython.display import clear_output
from matplotlib import pyplot
from api.sklearn_msgmlr import SklearnMSGMLR
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error
from helper.utils import colormap
def plot(model, thetas, x: str = None, y: str = None, ax = None, figsize = (14,8), show = True):
    """
    Plots the data points, model predictions, and uncertainty intervals.

    Input:
    - x: Name of the x-axis variable for plotting.
    - y: Name of the y-axis variable for plotting.
    - figsize: Tuple specifying the size of the plot.
    - truelabel: label of the true data points if available.

    Output: None
    """
    x = model.features_[0] if x is None else x
    y = model.target_[0] if y is None else y
    gammas, etas, betas, sigmas = model.unpack(thetas)
    if ax is None:
        fig, ax = pyplot.subplots(1, 1, figsize = figsize) 
    ls = []
    
    priors, postiors = model.update(model.X_, model.y_, thetas)
    smoothed = model.filter(model.X_, model.y_, thetas, priors, postiors)

    data_X = pd.DataFrame(model.X_, columns=model.features_)
    data_y = pd.DataFrame(model.y_, columns=model.target_)
    
    for g in range(model.n_clusters):
        ax.scatter(data_X[x][np.argmax(smoothed, axis=1) == g], 
                data_y[y][np.argmax(smoothed, axis=1) == g], 
                color = colormap[g], label = 'state '+str(g)+ ' data')
        xaxis = np.linspace(data_X[x].min() - 0.05*(data_X[x].max() - data_X[x].min()), 
                            data_X[x].max() + 0.05*(data_X[x].max() - data_X[x].min()), 10)
        otherxs = copy.deepcopy(model.features_)
        otherxs.remove(x)
        beta = pd.DataFrame(betas[g,:,:].T, index = model.target_, columns = model.features_)
        intercept = 0
        for otherx in otherxs:
            intercept += data_X[otherx].mean()*beta.loc[y, otherx]
        ls.append(ax.plot(xaxis, intercept + beta.loc[y, x]*xaxis, label = 'state '+str(g)+ ' model', color = colormap[g]))
        sigma = pd.DataFrame(sigmas[g,:,:], index = model.target_, columns = model.target_)
        ax.fill_between(xaxis, intercept + beta.loc[y, x]*xaxis + 1.96 * np.sqrt(sigma.loc[y,y]),
                        intercept + beta.loc[y, x]*xaxis - 1.96 * np.sqrt(sigma.loc[y,y]), alpha = 0.2, color = colormap[g])
        ax.fill_between(xaxis, intercept + beta.loc[y, x]*xaxis + 1.68 * np.sqrt(sigma.loc[y,y]),
                        intercept + beta.loc[y, x]*xaxis - 1.68 * np.sqrt(sigma.loc[y,y]), alpha = 0.4, color = colormap[g])
    ax.legend(frameon = False, fontsize = 12, loc = 'upper left', ncol = model.n_clusters)
    [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_title('Scatter ' + x + '-' + y, fontsize=14)
    if show: pyplot.show()
    return None if show else fig, ax

def plot_history(model, t = None, figsize = (14,8), ax = None, show = True):
    # plot the state graph
    t = range(model.X_.shape[0]) if t is None else t

    if ax is None:
        fig, ax = pyplot.subplots(model.n_clusters, 1, figsize = figsize)

    for g in range(model.n_clusters):
        ax[g].plot(t, model.smoothed_[:,g], color = colormap[g], label = 'Smoothed Probability')
        [ax[g].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax[g].tick_params(axis='both', which='major', labelsize=10)
        ax[g].set_ylabel('Probability', fontsize=12)
        ax[g].set_title('Smoothed Probability in State {}'.format(g), fontsize=14)
    
    if show: pyplot.show()
    return None if show else fig, ax
    

def plot_mse(model, y = None, figsize = (14,8), showci = True, alpha = 0.1, show = True):
    """
    Plots the true data points against predicted data points with Mean Squared Error information.

    Input:
    - y: Name of the dependent variable for plotting.
    - figsize: Tuple specifying the size of the plot.

    Output: None
    """
    if y is None:
        y = model.target_[0]
    data_y = pd.DataFrame(model.y_, columns=model.target_)
    ypred, stds = model.predict(X = None, pred_std=True)
    ypred = pd.DataFrame(ypred, columns=model.target_)
    stds = pd.DataFrame(stds, columns=model.target_)
    lbs = pd.DataFrame(ypred - stats.norm.ppf(1.0-alpha/2.0)*stds, columns=model.target_)
    ubs = pd.DataFrame(ypred + stats.norm.ppf(1.0-alpha/2.0)*stds, columns=model.target_)
    fig, ax = pyplot.subplots(1, 1, figsize = figsize)
    line45 = [min(model.y_.min(), ypred[y].min()), max(model.y_.max(), ypred[y].max())]
    ax.plot(line45, line45, '--', color = 'black')
    for g in range(model.n_clusters):
        priors, postiors = model.update(model.X_, model.y_, model.thetas_)
        smoothed = model.filter(model.X_, model.y_, model.thetas_, priors, postiors)
        lbg = lbs[y][np.argmax(smoothed, axis = 1) == g]
        ubg = ubs[y][np.argmax(smoothed, axis = 1) == g]
        avg = (lbg.values + ubg.values)/2
        if showci:
            ax.errorbar(x = data_y[y][np.argmax(smoothed, axis = 1) == g],
                        y = ypred[y][np.argmax(smoothed, axis = 1) == g],
                        yerr = np.vstack([avg.reshape((1, avg.size)) - lbg.values.reshape((1, lbg.size)), 
                                        ubg.values.reshape((1, ubg.size)) - avg.reshape((1, avg.size))]),
                        marker = 'o', color = colormap[g], label = 'state '+str(g)+ ' {:d}'.format(int(100*(1-alpha/2.0)))+ '%CI',
                        elinewidth=2, capsize=3, linewidth = 0)
        else:
            ax.scatter(data_y[y][np.argmax(smoothed, axis = 1) == g],
                        ypred[y][np.argmax(smoothed, axis = 1) == g], 
                        color = colormap[g], label = 'state '+str(g))
            
    [ax.spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(frameon = False, fontsize = 12, loc = 'upper left')
    ax.set_xlabel('True Data', fontsize=12)
    ax.set_ylabel('Pred Data', fontsize=12)
    ax.set_title('True Data vs Pred Data: ' + y + ', Mean Squared Error: {:.4f}'.format(mean_squared_error(data_y[y], ypred[y])), fontsize=14)
    if show: pyplot.show()
    return None if show else fig, ax

class MSGMLR(SklearnMSGMLR):
    def fit(self, X, y, plot_x: str = None, plot_y: str = None, figsize = (14,8)):
        """
        Fit the MSGMLR model to the provided data.

        Args:
        - X: Feature data matrix.
        - y: Target data matrix.
        - plot_x: Name of the x-axis variable for plotting (optional).
        - plot_y: Name of the y-axis variable for plotting (optional).
        - figsize: Tuple specifying the size of the plot.

        Output: None
        """
        if (not get_ipython()) and self.verbose > 2:
            pyplot.ion()
            fig, ax = pyplot.subplots(1, 1, figsize = (14,8))
            do_show = True
        else:
            pyplot.ioff()
            ax = None
            do_show = False

        def plot(thetas):
            if not get_ipython(): pyplot.cla()
            self.plot(thetas, plot_x, plot_y, ax = ax, figsize = figsize, show = do_show)
            clear_output(wait = True)
            pyplot.pause(0.1)
        self.step_plot = plot if self.verbose > 2 else None
        super().fit(X, y)

        if not get_ipython():
            pyplot.ioff()
            pyplot.show()
            pyplot.close()
        return self
    
    def plot(self, thetas, x: str = None, y: str = None, ax = None, figsize = (14,8), show = True):
        return plot(self, thetas, x, y, ax, figsize, show)
    
    def plot_history(self, t = None, figsize = (14,8), ax = None, show = True):
        return plot_history(self, t, figsize, ax, show)

    def plot_mse(self, y = None, figsize = (14,8), showci = True, alpha = 0.1, show = True):
        return plot_mse(self, y, figsize, showci, alpha, show)