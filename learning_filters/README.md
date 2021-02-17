# Learning Filters
This repo documents my notes taken during the learning of [Kalman and Bayesian Fitlers](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python).

## Overview
- Sensors are noisy. Results from algorithm are also noisy (cf perception algorithms). This is the motivation behind a huge body of work in filtering.
- Knowledge is uncertain, and we alter our beliefs based on the strength of the evidence. **Our principle is to never discard information.**
- Our beliefs depend on the past and on our knowledge of the system we are tracking and on the characteristics of the sensors.


## 01-g-h-Filter
- **Two sensors, even if one is less accurate than the other, is better than one.** A less accurate measurement can improve our knowledge over a more accurate measurement. 
- If we only form estimates from the measurement then the prediction will not affect the result. If we only form estimates from the prediction then the measurement will be ignored. If this is to work **we need to take some kind of blend of the prediction and measurement**.
- Math doesn't know if the data came from a measurement or a prediction, it only makes computations based on the value and accuracy of those values.
- Estimate always falls between the measurement and prediction.
- Data is better than a guess, even if it is noisy.
- The [g-h Filter](https://en.wikipedia.org/wiki/Alpha_beta_filter)
	- This algorithm is known as the g-h filter or the α-β filter. g and h refer to the two scaling factors that we used in our example. g is the scaling we used for the measurement (weight in our example), and h is the scaling for the change in measurement over time (lbs/day in our example). α and β are just different names used for this factors.
	- alpha beta filter 真正的内核所在则是：”估计 = 预测 + 测量“
	- 或者用更bayesian的话说：”posterior = prior + likelihood“
	- 或者更general一点说：Don't lose any information.

```python
weight = 160.  # initial guess
gain_rate = -1.0  # initial guess

time_step = 1.
weight_scale = 4./10
gain_scale = 1./3
estimates = [weight]
predictions = []

for z in weights:
    # prediction step
    weight = weight + gain_rate*time_step
    gain_rate = gain_rate  # constant growth rate model
    predictions.append(weight)
    
    # update step    
    residual = z - weight
    
    gain_rate = gain_rate + gain_scale   * (residual/time_step)
    weight    = weight    + weight_scale * residual
  
    estimates.append(weight)

gh.plot_gh_results(weights, estimates, predictions, [160, 172])
```
- Principles of filtering
	- Multiple data points are more accurate than one data point, so throw nothing away no matter how inaccurate it is.
	- Always choose a number part way between two data points to create a more accurate estimate.
	- Predict the next measurement and rate of change based on the current estimate and how much we think it will change.
	- The new estimate is then chosen as part way between the prediction and next measurement scaled by how accurate each is.
- Any estimation problem consists of forming an estimate of a hidden state via observable measurements.
- We use a **process model** to mathematically model the system.  The system error or process error is the error in this model. We never know this value exactly; if we did we could refine our model to have zero error. Some texts use plant model and plant error. You may also see system model. They all mean the same thing.
- The **predict step** is known as system propagation. It uses the process model to form a new state estimate. Because of the process error this estimate is imperfect. Assuming we are tracking data over time, we say we propagate the state into the future. Some texts call this the **evolution**.
- The **update step** is known as the measurement update. One iteration of the system propagation and measurement update is known as an epoch.
- Effect of varying g and h
	- With large g, the estimation follows measurement more. If measurement is more accurate, then choose a larger g.
	- A larger h will cause us to react more quickly and settle to the correct value faster if the initial guess is off. If the process model dictates that the object status changes slowly, then use a smaller h.
	- A filter that ignores measurements (g=0, h=0) is useless.
	- g and h must reflect the real world behavior of the system you are filtering, not the behavior of one specific data set. 
	- ad-hoc choices for g and h do not perform very well. We need a data-dependent strategy to update g and h. 
	- **Filters are designed, not selected.**

```
Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state
Predict

1. Use system behavior to predict state at the next time step
2. Adjust belief to account for the uncertainty in prediction
Update

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. New estimate is somewhere on the residual line
```

	
## 02-Discrete-Bayes
- Bayes filter is a special type of g-h filter. The discrete Bayes filter used a histogram of probabilities to track the state of a system.
- Updated knowledge = likelihood of new knowledge x prior knowledge.
- A **prior** is the probability prior to incorporating measurements or other information. In discrete Bayes filter, a prior is the predicted value based on posterior value from last step.
- In practice statisticians use a mix of frequentist and Bayesian techniques. Sometimes finding the prior is difficult or impossible, and frequentist techniques rule.
- Predict step always degrades our knowledge, but the addition of another measurement, even when it might have noise in it, improves our knowledge.
- The essence of Bayes filter is that multiplying probabilities when we measure, and shifting probabilities when we update leads to a converging solution
- [Likelihood](https://en.wikipedia.org/wiki/Likelihood_function): When we computed belief[hall==z] *= scale we were computing how likely each position was given the measurement. **The likelihood is not a probability distribution because it does not sum to one.**

```python
# predict: move posterior from next step by offset, and blurred by kernel due to noise. Predict always loses information.
prior = predict(posterior, offset=1, kernel=kernel)

# measurement corresponds to some likelihood
likelihood = lh_hallway(hallway, z=0, z_prob=.75)

# update: update likelihood by prior
posterior = update(likelihood, prior)
```
- Drawbacks of discrete Bayes filter:
	- Scaling
	- Discrete: too many bins in reality
	- Multimodal: not really a problem as particle filter is also multi-modal and is heavily used because of this trait. 
- For many tracking and filtering problems our desire is to have a filter that is **unimodal and continuous**. 

```
Initialization

1. Initialize our belief in the state
Predict

1. Based on the system behavior, predict state for the next time step
2. Adjust belief to account for the uncertainty in prediction
Update

1. Get a measurement and associated belief about its accuracy
2. Compute how likely it is the measurement matches each state
3. Update state belief with this likelihood
```

## 03-Gaussians
- Kalman filters use Gaussians because they are computationally nice: the result of adding or multiplying two Gaussians distributions is a Gaussian function
- Bayes theorem has two parts
	- Denominator is just a normalization factor
	- Bayes turns an intractable problem (given a measurement guess the state) into a tractable one (given a state, what is the measurement, scaled by the prior of the states)
- Bayes theorem can be seen as an **inverse problem solving** technique in engineering. It is used in other fields, such as inverse rendering, launch multiple models and use the one with the smallest error terms, etc. 
- Designers for mission critical filters, such as the filters on spacecraft, need to master a lot of theory and **empirical knowledge about the performance of the sensors** on their spacecraft. --> Same for Sensor Fusion engineers who need to know a lot about the object detector and their characterization.
- The t-distribution is symmetric and bell-shaped, like the normal distribution, but has heavier tails, meaning that it is more prone to producing values that fall far from its mean. The smaller the DoF ($\nu$) is, the more heavy-tailed the distribution is. As the number of degrees of freedom grows, the t-distribution approaches the normal distribution with mean 0 and variance 1. For this reason $\nu$ is also known as the normality parameter. 


## [04-One-Dimensional-Kalman-Filters](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/04-One-Dimensional-Kalman-Filters.ipynb)
- The Kalman filter is a Bayesian filter that uses Gaussians.
- The 1-D Kalman filter uses Gaussian to simplify the discrete Bayes filter.
- We can represent the prior with a Gaussian. What about the likelihood? The likelihood is the probability of the measurement given the current state.
- The variance of the product of two Gaussians. The variance is smaller than either one. This makes sense as two measurements are better than one.  $\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2}$

```
Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state
Predict

1. Use system behavior to predict state at the next time step
2. Adjust belief to account for the uncertainty in prediction
Update

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. Compute scaling factor based on whether the measurement
or prediction is more accurate
4. set state between the prediction and measurement based 
on scaling factor
5. update belief in the state based on how certain we are 
in the measurement
```
- Kalman gain: How much to trust the new measurement. The ratio between prior uncertainty vs measurement uncertainty. For example, if the measurement device is 10 times more accurate than the prior, then K=9/10.
- Convention:
	- z: measurement
	- R: measurement noise
	- u: movement due to process
	- Q: process noise
	- x: state
	- P: variance of state


```python
def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement
    
    y = z - x        # residual
    K = P / (P + R)  # Kalman gain

    x = x + K*y      # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)

def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)
```
- $$\mathcal{N}(\bar\mu,\, \bar\sigma^2) = \mathcal{N}(\mu,\, \sigma^2) + \mathcal{N}(\mu_\mathtt{move},\, \sigma^2_\mathtt{move})$$
- $$\mathcal{N}(\mu,\, \sigma^2) = \mathcal{N}(\bar\mu,\, \bar\sigma^2)  \times \mathcal{N}(\mu_\mathtt{z},\, \sigma^2_\mathtt{z})$$
- This **orthogonal projection** approach form obscures the Bayesian aspect of Kalman filter, but is the original approach Kalman filter got invented.
![](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/figs/residual_chart.png)
- Kalman filters under adverse conditions
	- Extreme noise: trust prediction more. Very smooth after filtering.
	- Incorrect process variance: process variance tells how much the system is changing. Math requires that the variances correctly describe your system. The key point is to recognize that math requires that the variances correctly describe your system. The filter does not 'notice' that it is diverging from the measurements and correct itself. It computes the Kalman gain from the variance of the prior and the measurement, and forms the estimate depending on which is more accurate.
	- Bad initialization: converges quickly as long as 
	- Extremely noisy data and extremely bad initial conditions: takes longer to converge
- Fixed Gain Filters
	- On embedded devices, computation resource is limited. Kalman gain usually converge to a fixed value.  
	- We do not need to consider the variances at all. If the variances converge to a single value so does the Kalman gain. 
	
```python
def update(x, z):
    K = .13232  # experimentally derived Kalman gain
    y = z - x   # residual
    x = x + K*y # posterior
    return x
    
def predict(x):
    return x + vel*dt
```	
- Prior is usually not mentioned in Kalman Filter literature. Both the prior and the posterior are the estimated state of the system, the former is the estimate before the measurement is incorporated, and the latter is after the measurement has been incorporated.


## [05-Multivariate-Gaussians](https://nbviewer.jupyter.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/05-Multivariate-Gaussians.ipynb)
- The product of two Gaussians are a taller Gaussian.
- Observed variable, Hidden variables, unobservable variable.
- **A multivariate Kalman filter can perform better than a univariate one.** Correlations between variables can significantly improve our estimates
- If we can express our uncertainties as a multidimensional Gaussian we can then multiply the prior with the likelihood and get a much more accurate result.
- Correlation between variables can drastically improve the posterior. If we only roughly know position and velocity, but they are correlated, then our new estimate can be very accurate.

