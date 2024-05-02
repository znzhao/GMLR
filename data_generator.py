# Import necessary modules from helper and standard libraries
from helper.utils import Timer
import numpy as np
import pandas as pd

# Define a class to generate panel data
class PanelGenerator:
    def __init__(self, nX=2, ny=2, ngroup=2, nobs=200, Xrange=(-10, 10), gammarange=(-1, 1), betarange=(-1, 1), sigmarange=(0.1, 0.5), seed = 0):
        '''
        Initialize the panel data generator with default or user-provided parameters.
        Parameters include the number of independent variables, dependent variables,
        groups, observations, and ranges for generating the gamma, beta, and sigma parameters.
        '''
        np.random.seed(seed)
        self.ny = ny  # Number of dependent variables
        self.nX = nX+1  # Number of independent variables including a constant term
        self.ngroup = ngroup  # Number of groups in the panel data
        self.nobs = nobs  # Number of observations
        # Generate column names for X and y variables
        self.Xcol = ['X'+str(i) for i in range(nX)]
        self.ycol = ['y'+str(i) for i in range(ny)]
        # Initialize X data frame with random values and add a constant column
        self.X = pd.DataFrame(Xrange[0] + (Xrange[1] - Xrange[0]) * np.random.rand(nobs, nX), columns=self.Xcol)
        self.X.loc[:, 'Const'] = 1  # Add constant column for intercept

        # Calculate the number of gamma, beta, and sigma parameters to be estimated
        self.ngammas = (self.ngroup-1)*self.nX
        self.nbetas = self.nX*self.ngroup*self.ny
        self.nsigmas = self.ngroup*self.ny
        self.ncovs = self.ngroup*int(self.ny*(self.ny-1)/2)

        # Initialize parameters gamma, beta, and sigma within specified ranges
        self.gammas = gammarange[0] + (gammarange[1] - gammarange[0]) * np.random.rand((self.ngroup-1)*self.nX)
        self.gammas = np.reshape(self.gammas, newshape=(self.ngroup-1, self.nX))
        self.betas = betarange[0] + (betarange[1] - betarange[0]) * np.random.rand(self.nX*self.ngroup*self.ny)
        self.betas = np.reshape(self.betas, newshape=(self.ngroup, self.nX, self.ny))
        self.sigmas = sigmarange[0] + (sigmarange[1] - sigmarange[0]) * np.random.rand(self.ny*self.ngroup)
        self.sigmas = np.reshape(self.sigmas, newshape=(self.ngroup, self.ny))

        # Simulate the dependent variables y based on the model and parameters
        self.y, self.g = self.simulate()
        self.y = pd.DataFrame(self.y, columns=self.ycol)
        self.g = pd.DataFrame(self.g, columns=['group'])
        # Combine the independent and dependent variables into a single data frame
        self.data = pd.concat([self.X[self.Xcol], self.y[self.ycol], self.g], axis=1)

    def simulate(self):
        '''
        Simulate the dependent variables (y) using the specified gamma, beta, and sigma parameters.
        This method generates the y values for each observation based on the model and adds noise accordingly.
        '''
        ys = []  # Initialize an empty list to store simulated y values
        gs = []  # Initialize an empty list to store simulated g values
        with Timer('Data Generation'):  # Use the Timer context manager to measure execution time
            for t in range(self.nobs):  # For each observation
                # Calculate group probabilities based on gamma parameters and logit model
                priors = np.exp(self.gammas.dot(self.X.iloc[t])) / (np.sum(np.exp(self.gammas.dot(self.X.iloc[t])))+1)
                threshold = np.cumsum(priors)  # Cumulative sum for group assignment
                z = np.random.rand()  # Random value for group assignment
                g = np.sum(z > threshold)  # Determine group based on threshold
                # Generate y values based on beta parameters and normal error term
                y = self.betas[g].T.dot(self.X.iloc[t]) + np.random.normal(scale=self.sigmas[g])
                ys.append(y)  # Append the generated y values to the list
                gs.append(g)  # Append the generated y values to the list
        return np.vstack(ys), np.vstack(gs)

    def summary(self):
        '''
        Print a summary of the simulation parameters, gamma, and beta coefficients.
        This provides an overview of the model specifications and the generated parameters.
        '''
        # Print header for the summary
        print('#' + '-'*91 + '#')
        print('{:^10s}'.format('Simulation Parameters').center(93))
        print('#' + '-'*91 + '#')
        # Print gamma coefficients
        print('Gamma: Logit Regression Coefficients')
        gammas = pd.DataFrame(self.gammas, index=['state '+str(x) for x in range(self.ngroup-1)], columns=self.Xcol + ['Const'])
        print(gammas.to_string(justify="center", col_space=int(74/self.nX)))
        print('='*93)
        # Print beta coefficients for each group
        print('Beta: Main Model Regression Coefficients')
        for g in range(self.ngroup):
            print('{:^10s}'.format('State ' + str(g)).center(93))
            betag = pd.DataFrame(self.betas[g, :, :].T, index=self.ycol, columns=self.Xcol + ['Const'])
            print(betag.T.to_string(justify="center", col_space=int(68/self.ny)))
        print('='*93)
        # Print sigma coefficients for each group
        print('Sigma: Error Coefficients')
        sigmas = pd.DataFrame(self.sigmas, index=['state '+str(x) for x in range(self.ngroup)], columns=self.ycol)
        print(sigmas.to_string(justify="center", col_space=int(68/self.ny)))
        print('='*93)

# Define a class to generate panel data
class TSGenerator:
    def __init__(self, nX=2, ny=2, ngroup=2, nobs=200, Xrange=(-1, 1), gammarange=(-1, 1), etarange = (-1, 1), betarange=(-1, 1), sigmarange=(0.1, 0.5), seed = 0):
        '''
        Initialize the panel data generator with default or user-provided parameters.
        Parameters include the number of independent variables, dependent variables,
        groups, observations, and ranges for generating the gamma, beta, and sigma parameters.
        '''
        np.random.seed(seed)
        self.ny = ny  # Number of dependent variables
        self.nX = nX+1  # Number of independent variables including a constant term
        self.ngroup = ngroup  # Number of groups in the panel data
        self.nobs = nobs  # Number of observations
        # Generate column names for X and y variables
        self.Xcol = ['X'+str(i) for i in range(nX)]
        self.ycol = ['y'+str(i) for i in range(ny)]
        # Initialize X data frame with random values and add a constant column
        self.X = pd.DataFrame(Xrange[0] + (Xrange[1] - Xrange[0]) * np.random.rand(nobs, nX), columns=self.Xcol)
        self.X.loc[:, 'Const'] = 1  # Add constant column for intercept

        # Calculate the number of gamma, beta, and sigma parameters to be estimated
        self.ngammas = (self.ngroup-1)*self.nX
        self.nbetas = self.nX*self.ngroup*self.ny
        self.nsigmas = self.ngroup*self.ny
        self.ncovs = self.ngroup*int(self.ny*(self.ny-1)/2)

        # Initialize parameters gamma, beta, and sigma within specified ranges
        self.gammas = gammarange[0] + (gammarange[1] - gammarange[0]) * np.random.rand((self.ngroup-1)*self.nX)
        self.gammas = np.reshape(self.gammas, newshape=(self.ngroup-1, self.nX))
        self.etas = etarange[0] + (etarange[1] - etarange[0]) * np.random.rand((self.ngroup-1))
        self.etas = np.reshape(self.etas, newshape=(self.ngroup-1))
        self.betas = betarange[0] + (betarange[1] - betarange[0]) * np.random.rand(self.nX*self.ngroup*self.ny)
        self.betas = np.reshape(self.betas, newshape=(self.ngroup, self.nX, self.ny))
        self.sigmas = sigmarange[0] + (sigmarange[1] - sigmarange[0]) * np.random.rand(self.ny*self.ngroup)
        self.sigmas = np.reshape(self.sigmas, newshape=(self.ngroup, self.ny))

        # Simulate the dependent variables y based on the model and parameters
        self.y, self.g = self.simulate()
        # Combine the independent and dependent variables into a single data frame
        self.data = pd.concat([self.X[self.Xcol], pd.DataFrame(self.y, columns=self.ycol)[self.ycol]], axis=1)

    def simulate(self):
        '''
        Simulate the dependent variables (y) using the specified gamma, beta, and sigma parameters.
        This method generates the y values for each observation based on the model and adds noise accordingly.
        '''
        ys = []  # Initialize an empty list to store simulated y values
        gs = []
        lastg = 0
        with Timer('Data Generation'):  # Use the Timer context manager to measure execution time
            for t in range(self.nobs):  # For each observation
                # Calculate group probabilities based on gamma parameters and logit model
                priors = np.exp(self.gammas.dot(self.X.iloc[t]) + self.etas.dot(np.array([lastg==g for g in range(self.ngroup-1)]))) 
                priors = priors / (np.sum(priors)+1)
                threshold = np.cumsum(priors)  # Cumulative sum for group assignment
                z = np.random.rand()  # Random value for group assignment
                g = np.sum(z > threshold)  # Determine group based on threshold
                # Generate y values based on beta parameters and normal error term
                y = self.betas[g].T.dot(self.X.iloc[t]) + np.random.normal(scale=self.sigmas[g])
                ys.append(y)  # Append the generated y values to the list
                lastg = g
                gs.append(g)
        return np.vstack(ys), np.vstack(gs)  # Return the y values as a numpy array

    def summary(self):
        '''
        Print a summary of the simulation parameters, gamma, and beta coefficients.
        This provides an overview of the model specifications and the generated parameters.
        '''
        # Print header for the summary
        print('#' + '-'*91 + '#')
        print('{:^10s}'.format('Simulation Parameters').center(93))
        print('#' + '-'*91 + '#')
        # Print gamma coefficients
        print('Gamma: Logit Regression Coefficients')
        gammas = pd.DataFrame(self.gammas, index=['state '+str(x) for x in range(self.ngroup-1)], columns=self.Xcol + ['Const'])
        print(gammas.to_string(justify="center", col_space=int(74/self.nX)))
        # Print gamma coefficients
        print('Eta: Logit Regression Coefficients')
        etas = pd.DataFrame(self.etas, index=['state '+str(x) for x in range(self.ngroup-1)])
        print(etas.to_string(justify="center", col_space=int(74/self.nX), header = False))
        print('='*93)
        # Print beta coefficients for each group
        print('Beta: Main Model Regression Coefficients')
        for g in range(self.ngroup):
            print('{:^10s}'.format('State ' + str(g)).center(93))
            betag = pd.DataFrame(self.betas[g, :, :].T, index=self.ycol, columns=self.Xcol + ['Const'])
            print(betag.T.to_string(justify="center", col_space=int(68/self.ny)))
        print('='*93)
        # Print sigma coefficients for each group
        print('Sigma: Error Coefficients')
        print('{:^10s}'.format('State ' + str(g)).center(93))
        sigmas = pd.DataFrame(self.sigmas, index=['state '+str(x) for x in range(self.ngroup)], columns=self.ycol)
        print(sigmas.to_string(justify="center", col_space=int(68/self.ny)))        
        print('='*93)

# Main block to execute the panel data generator and print the summary
if __name__ == "__main__":
    paneldata = TSGenerator(ngroup=3)  # Create an instance of the PanelGenerator
    paneldata.summary()  # Call the summary method to print the simulation parameters and coefficients
