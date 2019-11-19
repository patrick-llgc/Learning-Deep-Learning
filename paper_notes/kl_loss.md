# [KL Loss: Bounding Box Regression with Uncertainty for Accurate Object Detection](https://arxiv.org/abs/1809.08545)

_November 2019_

tl;dr: Predict bbox as two corners with mean and variance. 

#### Overall impression
Similar to [IoU Net](iou_net.md), classification confidence is not always strongly related to localization confidence.

The paper models KL divergence loss between a Dirac delta label and a Gaussian prediction. Essentially this is NLL loss. For a more generalized KL loss, see [LaserNet KL](lasernet_kl.md).

Variance voting is quite interesting idea and can be used even without variance scores (just down-weigh by IoU and weigh by confidence score). I am quite surprised this has not been tried before.

Learning localization confidence in addition to classification confidence can 1) give interpretable results 2) leads to more precise localization (AP90).

[KL Loss](kl_loss.md) and [IoU Net](iou_net.md) are similar, but are different in implmentation. KL loss directly regresses mean and var from the same head, instead of a separate head for IoU prediction in IoU Net. Also Var Voting is one forward pass, not like the IoU Net's iterative optimization.

#### Key ideas
- **KL loss**: assuming Gaussian and Dirac makes the problem more tractable and has a closed form. Same to NLL loss of a Gaussian distribution, it degenerates to a L2 loss when sigma is fixed. The authors further modifies the regression target by log transform and changed the loss to an uncertainty-aware smooth L1 loss. 
$$L_{reg} = \frac{e^{-\alpha}}{2}((x_g - x_e)^2 ) + \frac{1}{2} \alpha, when <1$$
$$L_{reg} = e^{-\alpha}(|x_g - x_e| - \frac{1}{2}) + \frac{1}{2} \alpha, when >=1$$
or $$L_{reg} = e^{-\alpha} SL1(x_g - x_e) + \alpha$$
- **Variance Voting**: all previous NMS scheme does not weight average all bounding box but rather select one and suppress others, including [IoU Guided NMS](iou_net.md) and soft NMS (only changes score). Boxes with higher variance and further away from the bbox of interest is down-weighted. The final bbox position is a weighted average of the bbox.  --> this is similar to the bayesian inference scheme in [BayesOD](bayes_od.md).


#### Technical details
- The learned variance through KL loss is interpretable. 
- SoftNMS can be stacked to KL loss. Var voting can also greatly boost the AP performance. It mainly comes from more accurate localization, as AP50 barely improves. 

#### Notes
- Questions and notes on how to improve/revise the current work  

