# [LaserNet KL: Learning an Uncertainty-Aware Object Detector for Autonomous Driving](https://arxiv.org/abs/1910.11375)

_November 2019_

tl;dr: Estimating uncertainty with KL divergence loss assuming noisy label and leads to better performance than NLL loss assuming perfect label.

#### Overall impression
Fomulation of bbox: corners, thus all predictions have equal weights. 

Assuming zero noise in label and using NLL to regress for uncertainty has undesirable properties such as infinite gradient around optimum. This leads to the examples with low error and low uncertainty having larger gradients than examples with high error and high uncertainty, and thus lead to numerical instability and overfitting. 

> This paper assumes each label itself is a distribution and we learn the model by minimizing the Kullback_leibler (KL) divergence between the predicted distribution and the label distribution. 

> It is important for an autonomous vehicle to understand the uncertainty in its detections due to limited sensor data so that it can plan accordingly, e.g., slowing down the vehicle to collect ore data. 

The KL divergence loss is more stable and improves the performance on less common objects.

The paper also proposed a heuristic way to estimate noise in label. --> But this seems problematic and contrived. 

#### Key ideas
- Aleatoric uncertainty can arise from sensor noise, incomplete data, class ambiguity and label noise. It can be modeled by making the outputs of a neural network probabilistic by predicting a probability than a point estimate (essentially one additional number to regress the variance). 
- NLL loss of a Laplace distribution is essentially a L1 loss attenuated by noise estimation $\sigma$.
$$L_{NLL}(x, \mu, b) = \frac{|x - \mu|}{b} + \log 2b$$
- KL loss between two Laplace distributions: 
$$L_{KL}(\mu_l, b_l, \mu_p, b_p) = \log \frac{b_p}{b_l} + \frac{b_l \exp(-\frac{\mu_l - \mu_p}{b_l}) + |\mu_l - \mu_p|}{b_p} - 1$$ 
- When $b_l = 0$, KL=NLL.

#### Technical details
- The paper tried two methods to estimate the noise in labels, $b_l$. One fixed, 0.05 m (5 cm) error, and one heuristic, assigning more uncertainty to bigger objects, with the IoU of the convex hull of all available points and the current label of interest for each tracklet. 
- Lidar in KITTI only cares about the front 90 field of view of the LiDAR and up to 70 meters. 
- The authors mentioned that the uncertainty makes the prediction well-calibrated. --> Need to revisit this after reading the calibration papers. 

#### Notes
- Questions and notes on how to improve/revise the current work  

