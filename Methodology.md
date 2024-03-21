# Gaussian Mixture Linear Regression
## Cross Sectional Data
### Model Description 

Let's consider a scenario where we have a set of independent variables $X$ and a set of corresponding dependent variables $y$. Suppose the total number of observation is $N$. The value of each dependent variable $y_i$ depends on the group it belongs to. Mathematically, we express this relationship as:

$$
y_{i} = \beta_g X_i+\epsilon_{g,i}\quad \text{ if } i \text{ is in group } g
$$

where $\epsilon_{g,i}\sim N(0, \Sigma_g)$. Denote the group label of the $i$-th data point as $z_i$, the group membership probability for the $i$-th observation being in group $g$ is given by:

$$
P(z_i = g) = P(i \text{ is in group } g) = \frac{exp(\gamma_g X_i)}{1 + \sum_{g=\{1,2,...,G-1\}} exp(\gamma_g X_i)}
$$

Here, $\beta_g$ represents the coefficient for group $g$, and $\epsilon_{g,i}$ is the error term for the $i$-th observation in group $g$.

For all groups except the last one, the probability of the $i$-th observation being in group $G$ is calculated as:

$$
P(z_i = G) = P(i \text{ is in group } G) = \frac{1}{1 + \sum_{g=\{1,2,...,G-1\}} exp(\gamma_g X_i)}
$$

The challenge lies in the fact that the group membership is unobservable to the observer. Therefore, we cannot definitively determine which group each data point belongs to. Consequently, it becomes challenging to systematically estimate the parameters of the model, denoted as $\theta = {vec(\beta_g), vec(\gamma_g), vec(\Sigma_g)}$.

There are two new contribution of this method. First, The prior probability of falling in group $g$ is explicitly modeled. The advantage of doing this is that it will help us understand what will make the data to fall into a certain group. Second, there are multiple dependent variables. The previous literature has considered situation where there are only one dependent variable. However, in real world we might care more than one variable at the same time and the interaction between these variables is modeled with the variance covariance matrix $\Sigma_g$ in different states.

### Methodology

#### Maximum Likelihood Estimation

Suppose in step m-1 we obtain a set of parameter estimition. Given this information, the expected log-likelihood function $E[logP(\theta|X, y)]$ by definition is defined as

$$
E[logP(y|X, \theta)|X, y, \theta_{m-1}] = \frac{1}{N}\sum_{i = 1}^N \sum_{g = 1}^{G} P(z_i = g| X_i, y_i, \theta_{m-1})log(P(z_i = g|X_i, \theta)P(y_i|X_i, z_i = g, \theta))
$$

where by Bayesian rule,

$$
P(z_i = g| X_i, y_i, \theta_{m-1}) = \frac{P(z_i = g|X_i, \theta_{m-1})P(y_i|X_i, z_i = g, \theta_{m-1})}{\sum_{g = \{1,2,...,G\}}P(z_i = g|X_i, \theta_{m-1})P(y_i|X_i, z_i = g, \theta_{m-1})}
$$

Notice that $P(z_i = g|X_i, \theta_{m-1})$ denotes the prior probability that the $i$-th data falls in group $g$, which comes from the multi-logit model, while $P(z_i = g| X_i, y_i, \theta_{m-1})$ denotes the posterior probability after we observe the data. $P(y_i|X_i, z_i = g, \theta)$ and $P(y_i|X_i, z_i = g, \theta_{m-1})$ are defined under the assumption that the error follows a joint normal distribution. 

The model is solved with the EM algorithm. The algorithm contains two steps: expectation step and maximum likelihood step. We start with an initial guess of $\theta$. Given the guess in step $m-1$, the optimal maximum likelihood estimator $\hat \theta_m$ will solve:

$$
\hat \theta_{m} = argmax_\theta\ E[logP(y|X, \theta)|X, y, \theta_{m-1}] + \alpha|| \theta_{m-1}||_p
$$

where the last term is the $L^P$ norm of $\theta$, and the parameter $\alpha$ controls the degree of $L^P$ regularization. The baseline model sets $\alpha = 0$ such that it won't adjust for the $L^P$ regularization, and hence it is the normal MLE estimator. The algorithm will continue with $\theta$ set to the newly estimated parameters until $\hat\theta_m = \hat\theta_{m-1}$.

#### Asymptotic Estimation Variance

By law of large number, the asymptotic variance of the estimator $\hat \theta$ is given by

$$
Var(\hat\theta) = \frac{1}{df}(\frac{1}{N}\sum_{i = 0}^N \frac{\partial E[logP(y|X, \theta)|X, y, \theta]}{\partial \theta}^T  \frac{\partial E[logP(y|X, \theta)|X, y, \theta]}{\partial \theta})^{-1}
$$

where $df$ denote the degree of freedom, which is the number of observation minus the number of estimated parameters.

#### Variance Covariance Matrix Estimation

Since there are more than one dependent variable, the variance covariance matrix have many undetermined variables. To pin down these parameters, we need to make sure that the variance covariance matrix is a positive definite matrix. By LU decomposition of positive definite matrices, we can set the parameters to be undetermined lower triangular matrix $U_g$, where all the diagonal elements are positive. Then, we have $\Sigma_g = U_g U_g^{T}$, and the standard error of the matrix will be obtained with delta method, i.e. $Var(\hat\Sigma_g) = \frac{\partial \Sigma_g }{\partial \theta}^T Var(\hat\theta)\frac{\partial \Sigma_g }{\partial \theta}$​.

#### Model Prediction

After the estimation we can use the estimated model to obtain out of sample predictions. Suppose $X_i$ is the $i$-th out of sample independent variable matrix. The optimal model prediction $\tilde y$ satisfies:

$$
\tilde y = argmax_y\ E[logP(y|X_i, \hat \theta)|X_i, \hat \theta] =argmax_y\ \sum_{g = 1}^G P(z_i = g|X_i, \hat\theta)log(P(z_i = g|X_i, \hat\theta)P(y|\theta|X_i, z_i = g, \hat\theta))
$$

Notice the key difference between the objective function here and the objective function in the estimation step. The probability used for calculating the expectation is the prior probability in the prediction, while the probability used before is the posterior probability after observing $y$. Since with prediction there is no observed $y$, so naturally we can only use the prior to calculate the expectation. The prediction standard error is also calculated numerically with the delta method, i.e. $Var(\tilde y) = \frac{\partial \tilde y }{\partial \theta}^T Var(\hat\theta)\frac{\partial \tilde y }{\partial \theta}$.
