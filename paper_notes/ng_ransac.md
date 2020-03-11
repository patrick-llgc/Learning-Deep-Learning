# [NG-RANSAC: Neural-Guided RANSAC: Learning Where to Sample Model Hypotheses](https://arxiv.org/abs/1905.04132)

_March 2020_

tl;dr: Learn the weight of samples (correspondence) for Ransac. 

#### Overall impression
Ransac is an algorithm for robust (insensitive to observation outlier) estimation of parameters. Generally the sampling weights of all observation is uniform. NG-Ransac upweighs correct/in-lier correspondence and downweighs incorrect/outlier correspondence. 

The method of telling whether a correspondence (point-pair) is an inlier or not was inspired by [learning good correspondence](learning_correspondence.md) (CVPR 2018) and then both inspired [KP2D](kp2d.md) (ICLR 2020) as a supervision signal. The basic idea is to feed into a PointNet-like structure point-pair candidate and predict the probability of it being an in-lier. --> This can directly benefit the radar pin-camera bbox data association problem. 

The paper has a lot of details and is perhaps worth another thorough read.

#### Key ideas
- The formulation of NG-ransac facilitates training with any non-differentiable task loss function, and any non-differentiable model parameter solver, making it broadly applicable.
- The basic idea of the trick is to formulate the problem as a function of a probability distribution and minimize the expected task loss during training. Add a probabilistic wrapper around a deterministic algorithm to make it partially differentiable.

#### Technical details
- The idea can be also used for horizon line estimation and camera relocalization.

#### Notes
- [Official Github repo based on pytorch](https://github.com/vislearn/ngransac). --> Note that deploy this on Ubuntu. It runs into problems on Mac.

