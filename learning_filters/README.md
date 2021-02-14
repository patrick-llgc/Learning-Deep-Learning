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

	
## 02-Discrete-Bayes
- A **prior** is the probability prior to incorporating measurements or other information.
- ds