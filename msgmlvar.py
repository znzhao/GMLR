import numpy as np
import scipy.stats as stats
from IPython import get_ipython
from matplotlib import pyplot
from IPython.display import clear_output
from api.sklearn_msgmlvar import SklearnMSGMLVAR
from sklearn.utils.validation import check_is_fitted
from helper.utils import colormap
from msgmlr import plot, plot_history, plot_mse

class MSGMLVAR(SklearnMSGMLVAR):
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
        check_is_fitted(self)
        return plot_history(self, t, figsize, ax, show)
        
    def plot_mse(self, y = None, figsize = (14,8), showci = True, alpha = 0.1, show = True):
        check_is_fitted(self)
        return plot_mse(self, y, figsize, showci, alpha, show)
    
    def plot_irf(self, periods = 5, figsize = (14,8), show_ci = True, alpha = 0.1, show = True):
        """
        Plot impulse response functions (IRFs).

        Args:
            periods (int): Number of periods.
            figsize (tuple): Figure size.
            showci (bool): Whether to show confidence intervals.
            alpha (float): Confidence level.
            show (bool): Whether to display the plot.
        """
        check_is_fitted(self)
        fig, ax = pyplot.subplots(self.n_y_, self.n_y_, figsize = figsize)
        t_axis = range(periods)
        for state in range(self.n_clusters):
            irf = self.irf(state, periods)
            if show_ci:
                stds = self.irf_std(state, periods)
                lbs = irf - stats.norm.ppf(1.0-alpha/2.0)*stds
                ubs = irf + stats.norm.ppf(1.0-alpha/2.0)*stds
            for shock_i in range(self.n_y_):
                for y_i in range(self.n_y_):
                    ax[shock_i, y_i].plot(t_axis, irf[:, shock_i, y_i], color = colormap[state], label = 'state '+str(state))
                    if show_ci:
                        ax[shock_i, y_i].fill_between(t_axis, ubs[:, shock_i, y_i], lbs[:, shock_i, y_i], alpha = 0.4, color = colormap[state])

        for shock_i in range(self.n_y_):
            for y_i in range(self.n_y_):
                ax[shock_i, y_i].plot(t_axis, np.zeros(periods), '--', color = 'black')
                fig.tight_layout()
                [ax[shock_i, y_i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
                ax[shock_i, y_i].tick_params(axis='both', which='major', labelsize=10)
                ax[shock_i, y_i].legend(frameon = False, fontsize = 12, loc = 'upper left')
                ax[shock_i, y_i].set_xlabel('Periods', fontsize=12)
                ax[shock_i, y_i].set_ylabel('Unit', fontsize=12)
                ax[shock_i, y_i].set_title(self.target_[shock_i] + ' Shock -> ' + self.target_[y_i], fontsize=14)
                ax[shock_i, y_i].set_xticks(t_axis)
        if show: pyplot.show()
        return None if show else fig, ax
