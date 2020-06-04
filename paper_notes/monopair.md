# [MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships](https://arxiv.org/abs/2003.00504)

_June 2020_

tl;dr: mono3D with pair wise relation and non-linear optimization.

#### Overall impression
This work is inspired by [CenterNet](centernet.md). it not only predicts the 3d bbox from the center of the bbox (similar to [RTM3D](RTM3D) but without predicting the eight points directly). It is similar to the popular solutions to the [Kaggle mono3D competition](https://www.kaggle.com/c/pku-autonomous-driving).

The main idea is to predict distance of each instance and relative distance between neighboring pairs, and their corresponding uncertainties, then use nonlinear optimization (with g2o) for joint optimization. It refines the detection results based on spatial relationships

MonoPair improved accuracy dramatically, especially for heavily occluded scenario.

#### Key ideas
- Range circle: diameter is set up 
- Predicting relative distance is in local coordinate. This is a brilliant idea as this makes the regression target to be invariant to global azimuth. Regression target is multiplied by the rotational matrix of the azimuth angle.
- Predict uncertainty helps depth estimation greatly, as shown in Table 5.
- The joint optimization does not lead to too much improvement as shown in Table 6.


#### Technical details
- Regressing depth target $z = 1-\sigma(\hat{z})-1$
- Weight matrix is diagonal of predicted uncertainties of diff bits. $W = \text{diag}(1/\sigma_i) $. The authors tried various weighting strategies but no improvement.
- For images with more pair constraints, the performnace is better, even before 
- The addition of uncertainty to depth leads to the biggest improvement. 
- Almost real-time, at 57 ms per frame. This is partially due to the implementation of nonlinear optimization in g2o.

#### Notes
- [Demo](https://sites.google.com/view/chen3dmonopair)
- How much does baseline benefit from regressing the distance between two objects?
- The paper does not tell the details of how to match the edge prediction and vertices prediction.
