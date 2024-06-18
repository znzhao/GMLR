import numpy as np
from helper.utils import colormap
from matplotlib import pyplot
from scipy.stats import multivariate_normal
class CombinedDistr:
    def __init__(self, priors, means, covs, transformer = None):
        """
        Initialize the CombinedDistr class.

        Args:
            priors (np.array): Prior probabilities of each cluster.
            means (np.array): Means of the multivariate normal distributions for each cluster.
            covs (np.array): Covariance matrices of the multivariate normal distributions for each cluster.
            transformer (object or None): Optional transformer object for data transformation.

        Attributes:
            n_cluster (int): Number of clusters.
            priors (np.array): Prior probabilities of each cluster.
            means (np.array): Means of the multivariate normal distributions for each cluster.
            covs (np.array): Covariance matrices of the multivariate normal distributions for each cluster.
            transformer (object or None): Transformer object for data transformation.
            n_dim (int): Number of dimensions in the means array.
        """
        self.n_cluster = means.shape[-1]
        self.priors = priors
        self.means = means
        self.covs = covs
        self.transformer = transformer
        self.n_dim = means.ndim

    def check_dim(self):
        """
        Check if the dimension of means is 3D. Raise an error if it is.
        """
        if self.n_dim == 3:
            raise ValueError("Please specify which datapoint to look at.")
        
    def __getitem__(self, slice):
        """
        Slice the priors, means, and covs arrays and return a new CombinedDistr object.

        Args:
            slice: Slice object specifying indices or boolean mask.

        Returns:
            CombinedDistr: New CombinedDistr object with sliced attributes.
        """
        return CombinedDistr(self.priors[slice], self.means[slice], self.covs, self.transformer)

    def pdf(self, x):
        """
        Calculate the probability density function (PDF) of the combined distribution at point x.

        Args:
            x (np.array): Point(s) at which to evaluate the PDF.

        Returns:
            float: Probability density at point x.
        """
        self.check_dim()
        if self.transformer is None:
            res = 0
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.pdf(x, mean=self.means[:, g], cov=self.covs[g, :, :])
            return res
        else:
            res = 0
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.pdf(self.transformer.transform(x), mean=self.means[:, g], cov=self.covs[g, :, :])
            return res
        
    def cdf(self, x):
        """
        Calculate the cumulative distribution function (CDF) of the combined distribution at point x.

        Args:
            x (np.array): Point(s) at which to evaluate the CDF.

        Returns:
            float: Cumulative probability at point x.
        """
        self.check_dim()
        if self.transformer is None:
            res = 0
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.cdf(x, mean=self.means[:, g], cov=self.covs[g, :, :])
            return res
        else:
            res = 0
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.cdf(self.transformer.transform(x), mean=self.means[:, g], cov=self.covs[g, :, :])
            return res

    def marg_pdf(self, x, margin = 0):
        """
        Calculate the marginal probability density function (PDF) at point x along a specific margin.

        Args:
            x (int, float, or np.array): Point(s) at which to evaluate the marginal PDF.
            margin (int): Index of the dimension along which to compute the marginal PDF.

        Returns:
            float: Marginal probability density at point x along the specified margin.
        """
        self.check_dim()
        if self.transformer is None:
            res = 0
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.pdf(x, mean=[self.means[margin, g]], cov=[self.covs[g, margin, margin]])
            return res
        else:
            res = 0
            if type(x) == int or type(x) == float:
                x = np.array([x])
            x = self.transformer.transform(np.array([x]*self.means.shape[0]).T).T[margin,:]
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.pdf(x, mean=[self.means[margin, g]], cov=[self.covs[g, margin, margin]])
            return res

    def marg_cdf(self, x, margin = 0):
        """
        Calculate the marginal cumulative distribution function (CDF) at point x along a specific margin.

        Args:
            x (int, float, or np.array): Point(s) at which to evaluate the marginal CDF.
            margin (int): Index of the dimension along which to compute the marginal CDF.

        Returns:
            float: Marginal cumulative probability at point x along the specified margin.
        """
        self.check_dim()
        if self.transformer is None:
            res = 0
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.cdf(x, mean=[self.means[margin, g]], cov=[self.covs[g, margin, margin]])
            return res
        else:
            res = 0
            if type(x) == int or type(x) == float:
                x = np.array([x])
            x = self.transformer.transform(np.array([x]*self.means.shape[0]).T).T[margin,:]
            for g in range(self.n_cluster):
                res += self.priors[g] * multivariate_normal.cdf(x, mean=[self.means[margin, g]], cov=[self.covs[g, margin, margin]])
            return res
    
    def plot(self, margin = 0, threshold = 0, ax = None, figsize = (14,8), suptitle = None, show = True):
        """
        Plot various aspects of the combined distribution.

        Args:
            margin (int): Index of the dimension along which to plot.
            threshold (float): Threshold value for plotting.
            ax (matplotlib.axes.Axes or None): Optional matplotlib axes to plot on.
            figsize (tuple): Figure size (width, height) in inches.
            suptitle (str or None): Optional super title for the plot.
            show (bool): Whether to display the plot.

        Returns:
            matplotlib.axes.Axes: Axes object containing the plot.
        """
        if ax is None:
            fig, ax = pyplot.subplots(1, 2, figsize = figsize, width_ratios=[1, 3])
        ax[0].bar(x = ['state {}'.format(g) for g in range(self.n_cluster)], height = self.priors, color = [colormap[g] for g in range(self.n_cluster)])
        ax[0].axis('tight')
        [ax[0].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax[0].tick_params(axis='both', which='major', labelsize=10)
        ax[0].set_ylabel('Probability', fontsize=12)
        ax[0].set_title('State Probability', fontsize=14)
        
        std = np.sum([np.sum(np.sqrt(np.diag(self.covs[g,:,:]))) for g in range(self.n_cluster)])
        if self.transformer is None:
            x = np.linspace(np.min(self.means[margin, :]) - std, np.max(self.means[margin, :]) + std, 200)
        else:
            x_min = np.min(self.means[margin, :]) - std
            x_max = np.max(self.means[margin, :]) + std
            x_min = np.ones((1, self.means.shape[0]))*x_min
            x_max = np.ones((1, self.means.shape[0]))*x_max
            x = np.linspace(self.transformer.inverse_transform(x_min)[0, margin],
                            self.transformer.inverse_transform(x_max)[0, margin], 200)
        
        for g in range(self.n_cluster):
            if self.transformer is None:
                ax[1].axvspan(self.means[margin, g] - 1.68*np.sqrt(self.covs[g, margin, margin]), 
                              self.means[margin, g] + 1.68*np.sqrt(self.covs[g, margin, margin]), 
                              alpha=0.4, color=colormap[g])
            else:
                x_min = self.means[margin, g] - 1.68*np.sqrt(self.covs[g, margin, margin])
                x_max = self.means[margin, g] + 1.68*np.sqrt(self.covs[g, margin, margin])
                x_min = np.ones((1, self.means.shape[0]))*x_min
                x_max = np.ones((1, self.means.shape[0]))*x_max
                ax[1].axvspan(self.transformer.inverse_transform(x_min)[0, margin], 
                              self.transformer.inverse_transform(x_max)[0, margin], 
                              alpha=0.4, color=colormap[g])

        ax[1].plot(x, self.marg_pdf(x, margin), "grey")
        if self.transformer is None:
            ax[1].plot(x, self.marg_pdf(x, margin), "grey")
            ax[1].scatter(x = self.means[margin, :], 
                        y = self.marg_pdf(self.means[margin, :], margin), 
                        marker = 'o', 
                        color = [colormap[g] for g in range(self.n_cluster)])
        else:
            ax[1].scatter(x = self.transformer.inverse_transform(self.means.T).T[margin, :], 
                        y = self.marg_pdf(self.transformer.inverse_transform(self.means.T).T[margin, :], margin), 
                        marker = 'o', 
                        color = [colormap[g] for g in range(self.n_cluster)])
        ax[1].axvline(x = threshold, color="black", linestyle = '--')
        ax[1].axhline(y = 0, color="black", linestyle = '-')
        if self.transformer is None:
            ax[1].vlines(self.means[margin, :], 0, self.marg_pdf(self.means[margin, :], margin), 
                         linestyle="dashed", color = [colormap[g] for g in range(self.n_cluster)])
            ax[1].hlines(self.marg_pdf(self.means[margin, :], margin), threshold, self.means[margin, :], 
                         linestyle="dashed", color = [colormap[g] for g in range(self.n_cluster)])
        else:
            ax[1].vlines(self.transformer.inverse_transform(self.means.T).T[margin, :], 0, 
                         self.marg_pdf(self.transformer.inverse_transform(self.means.T).T[margin, :], margin), 
                         linestyle="dashed", color = [colormap[g] for g in range(self.n_cluster)])
            ax[1].hlines(self.marg_pdf(self.transformer.inverse_transform(self.means.T).T[margin, :], margin), threshold, 
                         self.transformer.inverse_transform(self.means.T).T[margin, :], 
                         linestyle="dashed", color = [colormap[g] for g in range(self.n_cluster)])
            
        ax[1].axis('tight')
        [ax[1].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        ax[1].tick_params(axis='both', which='major', labelsize=10)
        ax[1].set_xlabel('Predicted Value', fontsize=12)
        ax[1].set_ylabel('Prob Density', fontsize=12)
        if self.transformer is None:
            ax[1].set_title('Prob Density Function P(Value<{:.2f}) = {:.2f}'.format(threshold, self.marg_cdf(threshold, margin)), fontsize=14)
        else:
            
            ax[1].set_title('Prob Density Function P(Value<{:.2f}) = {:.2f}'.format(threshold, self.marg_cdf(threshold, margin)), fontsize=14)
        fig.suptitle(suptitle, fontsize=18)
        if show: pyplot.show()
        return None if show else fig, ax

