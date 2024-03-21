# Gaussian Mixture Linear Regression

## Model Description

Let's consider a scenario where we have a set of independent variables $X$ and a set of corresponding dependent variables $y$. Suppose the total number of observation is $N$. The value of each dependent variable $y_i$ depends on the group it belongs to. Mathematically, we express this relationship as:

$$
y_{i} = \beta_g X_i+\epsilon_{g,i}\quad \text{ if } i \text{ is in group } g
$$

where $\epsilon_{g,i}\sim N(0, \Sigma_g)$. Denote the group label of the $i$-th data point as $z_i$, the group membership probability for the $i$-th observation being in group $g$ is given by:

$$
P(z_i = g) = P(i \text{ is in group } g) = \frac{exp(\gamma_g X_i)}{1+\sum_{g = 1}^{G-1}exp(\gamma_g X_i)}
$$

Here, $\beta_g$ represents the coefficient for group $g$, and $\epsilon_{g,i}$ is the error term for the $i$-th observation in group $g$.

For all groups except the last one, the probability of the $i$-th observation being in group $G$ is calculated as:

$$
P(z_i = G) = P(i \text{ is in group } G) = \frac{1}{1+\sum_{g = 1}^{G-1}exp(\gamma_g X_i)}
$$

The challenge lies in the fact that the group membership is unobservable to the observer. Therefore, we cannot definitively determine which group each data point belongs to. Consequently, it becomes challenging to systematically estimate the parameters of the model, denoted as $\theta = {vec(\beta_g), vec(\gamma_g), vec(\Sigma_g)}$.

There are two new contribution of this method. First, The prior probability of falling in group $g$ is explicitly modeled. The advantage of doing this is that it will help us understand what will make the data to fall into a certain group. Second, there are multiple dependent variables. The previous literature has considered situation where there are only one dependent variable. However, in real world we might care more than one variable at the same time and the interaction between these variables is modeled with the variance covariance matrix $\Sigma_g$ in different states.

## Methodology

### Maximum Likelihood Estimation

The log-likelihood function $E[logP(\theta|X, y)]$ by definition is defined as

$$
E[logP(X, y, \theta)|X, y] = \frac{1}{N}\sum_{i = 1}^N \sum_{g = 1}^G P(\theta, z_i = g| X_i, y_i)log(P(z_i = g)P(\theta|X_i, y_i, z_i = g))
$$

where by Bayesian rule,

$$
P(\theta, z_i = g| X_i, y_i) = \frac{P(z_i = g)P(\theta|X_i, y_i, z_i = g)}{\sum_g P(z_i = g)P(\theta|X_i, y_i, z_i = g)}
$$

Notice that $P(z_i = g)$ denotes the prior probability that the $i$-th data falls in group $g$, which comes from the multi-logit model. $P(\theta, z_i = g| X_i, y_i)$ denotes the posterior probability after we observe the data. $P(\theta|X_i, y_i, z_i = g)$ is defined under the consumption that the error follows a joint normal distribution. The model is solved with the EM algorithm. The algorithm contains two steps: expectation step and maximum likelihood step. We start with an initial guess of $\theta$. Given the guess in step $t-1$, the optimal maximum likelihood estimator $\hat \theta_t$ will solve:

$$
\hat \theta_{t} = argmax_\theta\ E[logP(X, y, \theta_{t-1})|X, y] + \alpha|| \theta_{t-1}||_p
$$

where the last term is the $L^P$ norm of $\theta$, and the parameter $\alpha$ controls the degree of $L^P$ regularization. The baseline model sets $\alpha = 0$ such that it won't adjust for the $L^P$ regularization, and hence it is the normal MLE estimator. The algorithm will continue with $\theta$ set to the newly estimated parameters until $\hat\theta_t = \hat\theta_{t-1}$.

### Asymptotic Estimation Variance

By law of large number, the asymptotic variance of the estimator $\hat \theta$ is given by

$$
Var(\hat\theta) = \frac{1}{df}(\frac{1}{N}\sum_{i = 0}^N\frac{\partial E[logP(X_i, y_i, \theta)|X, y]}{\partial \theta}^T  \frac{\partial E[logP(X_i, y_i, \theta)|X, y]}{\partial \theta})^{-1}
$$

where $df$ denote the degree of freedom, which is the number of observation minus the number of estimated parameters.

### Variance Covariance Matrix Estimation

Since there are more than one dependent variable, the variance covariance matrix have many undetermined variables. To pin down these parameters, we need to make sure that the variance covariance matrix is a positive definite matrix. By LU decomposition of positive definite matrices, we can set the parameters to be undetermined lower triangular matrix $U_g$, where all the diagonal elements are positive. Then, we have $\Sigma_g = U_g U_g^{T}$, and the standard error of the matrix will be obtained with delta method, i.e. $Var(\hat\Sigma_g) = \frac{\partial \Sigma_g }{\partial \theta}^T Var(\hat\theta)\frac{\partial \Sigma_g }{\partial \theta}$​.

### Model Prediction

After the estimation we can use the estimated model to obtain out of sample predictions. Suppose $X_i$ is the $i$-th out of sample independent variable matrix. The optimal model prediction $\tilde y$ satisfies:

$$
\tilde y = argmax_t\ E[logP(X_i, y, \hat \theta)|X_i] = \sum_{g = 1}^G P(z_i = g) log(P(z_i = g)P(\theta|X_i, y_i, z_i = g))
$$

Notice the key difference between the objective function here and the objective function in the estimation step. The probability used for calculating the expectation is the prior probability in the prediction, while the probability used before is the posterior probability after observing $y$. Since with prediction there is no observed $y$, so naturally we can only use the prior to calculate the expectation. The prediction standard error is also calculated numerically with the delta method, i.e.$ Var(\tilde y) = \frac{\partial \tilde y }{\partial \theta}^T Var(\hat\theta)\frac{\partial \tilde y }{\partial \theta}$.

## State Space Gaussian Mixture Linear Regression

Let's consider a scenario where we have a sequence of independent variables $X_t$ and a set of corresponding dependent variables $y_t$. Suppose the total number of observed periods is $T$. The value of each dependent variable $y_t$ depends on the group it belongs to. Mathematically, we express this relationship as:

$$
y_{t} = \beta_g X_t+\epsilon_{g,t}\quad \text{ if } t \text{ is in state } g
$$

where $\epsilon_{g,t}\sim N(0, \Sigma_g)$. Denote the group label of time $t$ as $z_t$, the group membership probability for time $t$ being in group $g$ is given by:

$$
P(z_t = g|P(z_{t-1} = x)\ \forall x\in G) = \frac{exp(\gamma_g X_t + \sum_{x = 1}^{G}\eta_{g,x} P(z_{t-1} = x))}{1+\sum_{g = 1}^{G-1}exp(\gamma_g X_t + \sum_{x = 1}^{G}\eta_{g,x} P(z_{t-1} = x)))}
$$

Here, $\beta_g$ represents the coefficient for group $g$, and $\epsilon_{g,t}$ is the error term for time $t$ in group $g$.

For all groups except the last one, the probability of time $t$ being in group $G$ is calculated as:

$$
P(z_t = G|P(z_{t-1} = x)\ \forall x\in G) = \frac{1}{1+\sum_{g = 1}^{G-1}exp(\gamma_g X_t + \sum_{x = 1}^{G}\eta_{g,x} P(z_{t-1} = x)))}
$$

Notice that unlike the cross sectional data case, in the time series state space model $X_t$ doesn't have a constant in order to adjust for the fact that the total probability of the last period being in any group is 1.

### Algorithm

We start with a random guess of the state of the periods. The algorithm runs repeatedly the following 2 steps:

1. Estimation Step: given the probability of last period being in group $x$,i.e. the probability calculated in the update step for last period $t-1$, and the last step estimated parameters $\theta$, obtain the probability that this period falls in group $g$, i.e. $P(z_t = g|P(z_{t-1} = x)\ \forall x\in G, X_t, y_{t-1})$.

2. Update Step: update the probability that after observing $y_t$, the probability that period t falls in group $g$ according to Bayes rule, i.e. $P(z_t = G|P(z_{t-1} = x)\ \forall x\in G, X_t, y_t)$. Use this probability to update the probability for the next period: the update method is:


   $$
   P(z_t = G|P(z_{t-1} = x)\ \forall x\in G, X_t, y_t) = \frac{\phi(y_t - \beta_g X_t, \Sigma_g)P(z_t = g|P(z_{t-1} = x)\ \forall x\in G, X_t, y_{t-1})}{\sum_g \phi(y_t - \beta_g X_t, \Sigma_g)P(z_t = g|P(z_{t-1} = x)\ \forall x\in G, X_t, y_{t-1})}
   $$
   
   Use these to calculate the log-likelihood function:

   

4. Filtering Step: given the estimation of the parameters and the last period state, backout the probability that the current state falls in group g. 

