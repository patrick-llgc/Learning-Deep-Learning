# [ForeSeE: Task-Aware Monocular Depth Estimation for 3D Object Detection](https://arxiv.org/abs/1909.07701)

_October 2019_

tl;dr: Train a depth estimator focused on the foreground moving object and improve 3DOD based on pseudo-lidar.

#### Overall impression
This paper succeeds the line of work in pseudo-lidar ([pseudo-lidar](pseudo_lidar.md), [pseudo-lidar++](pseudo_lidar++.md), [pseudo-lidar e2e](pseudo_lidar_e2e.md)). 

Two overall issue with pseudo lidar idea: 1) inaccuracies in depth estimation and 2) blurry edges in depth map leading to edge bleeding. Like [pseudo-lidar e2e](pseudo_lidar_e2e.md), ForeSeE also realizes the drawbacks of using an off-the-shelf depth estimator, but instead of finetuning it end-to-end, it focuses on the more important foreground moving objects for 3DOD. 

The paper has a good introduction and background session. 

However the model seems to have much lower performance (even lower than pseudo-lidar). Email sent to authors to inquire about this. --> This turned out to be a game changer for pseudo-lidar:


**Note that DORN's training data overlaps with object detection's validation data, and suffers from overfitting.** Both [pseudo lidar](pseudo_lidar.md) and [pseudo lidar e2e](pseudo_lidar_e2e.md) suffer from this problem. According to the [ForeSeE](foresee_mono3dod.md) paper, if the validation data is excluded from the training of depth map, then PL's performance drops from 18.5 to 5.4 AP_3D at IoU=0.7.

![](../assets/images/pseudo_lidar.png)
![](../assets/images/foresee.png)

#### Key ideas
- Not all pixels are equal. This is particularly true for 3DOD. 

> Estimation error on a car is much different from the same error on a building.

- FG pixels tend to gather in clusters. 
- Depth estimation in bin classification. Depth values are discretized into **100 discrete bins in the log space**, instead of directly regression.
- Training with binary mask, weighted sum of L_fg + L_bg. 
- During inference, element-wise maximum value of confidence vector in C depth bins are obtained, and pass through a softmax. 

#### Technical details
- Foreground objects are more "bumpy" (non-zero Laplacian, 2nd order derivatives)

#### Notes
- Why the weight of both branches set to 0.2? I would expect larger than 0.5.
- Why the performance is so low (lower than the original PL paper)

