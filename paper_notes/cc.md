# [CC: Competitive Collaboration: Joint Unsupervised Learning of Depth, Camera Motion, Optical Flow and Motion Segmentation](https://arxiv.org/abs/1805.09806)

_August 2020_

tl;dr: Train a motion segmentation network to moderate the data feed into depth predictor and optical flow estimator. 

#### Overall impression
The paper is along the lines of [GeoNet](geonet.md) and [GLNet](glnet.md) by combining depth prediction and optical flow prediction. However the paper is articulated that the two are connected by a third task: **motion segmentation**. 

However [CC](cc.md) uses purely low level geometric constraints for self-supervised depth prediction, and cannot solve infinite depth issue by design.  The idea is taken further by [SGDepth](sgdepth.md), which incorporates semantic information into the task of motion segmentation. 

The overall training strategy is very complex, into 6-steps. The model's performance has been surpasssed by new techniques such as [SGDepth](sgdepth.md).

In [GeoNet](geone.md), optical flow network is used to predict "residual" flow and thus no coupling between depth and optical flow. This cascaded design prevents exploitation of inter-task dependencies. [DFNet](dfnet.md) exploit consistency between depth and flow, but did not account for moving objects.

Not all data in the unlabeled training set will conform to the SfMLeaner's assumption, and some of it may corrupt training. Thus one key question is to how to exclude such data, such as independently moving area. 

#### Key ideas
- Motion segmentation allow the network to use geometric constraints where they apply and generic flow there they do not. 
- R and F compete for training data (pixels), and the moderator is M (motion segmentation). R and F can also collaborate to train M. 
	- R = D (depth) + C (camera pose)
	- F (optical flow)
	- M comes from the idea of [Neural Expectation maximization](https://arxiv.org/abs/1708.03498), where one model is trained to distribute data to other models. 
	- The idea of a neural moderator is demonstrated by training a model to distinguish domain on a mixture of MNIST and SVHN datasets.
- Iterative training
	- In the first stage, moderator is fixed, and optimize depth and flow networks.
	- In the second stage, train moderator with the consensus of depth and flow.
- Losses
	- Reconstruction loss, moderated by static mask $m_s$
	- Flow loss, moderated by $(1-m_s)$
	- Mask loss, encourages the whole scene to be static, $m_s=1$. The distance is measured by CE.
	- Consensus loss, $m_s = I(R_{loss} < F_{loss}) \cup  I(\text{static scene flow = optical flow})$. Encourages a pixel to be labeled as static if R has a lower photometric error than F or if the induced flow of R is similar to that of F.
	- Smooth loss, encourages smooth depth, flow and motion mask. 
- Inference:
	- d and c are directly estimated from networks.
	- m is estimated by taking union (or intersection?) of predicted m and consensus between static scene flow and optical flow. This is the **motion segmentation** results.
	- f: full optical flow is computed as m * reprojected static scene flow + (1-m) * optical flow.

#### Technical details
- No sensor can provide GT for optical flow.
- Training strategy.
	- train D and C (sfm learner) with R loss
	- train F (optical flow) with F loss
	- train M with R and F loss
	- then loop over
		- competition
			- train D and C with R, F and M loss
			- train F with F and M loss
		- collaboration
			- train M with R, F, M and C loss

#### Notes
- [Github code](https://github.com/anuragranj/cc)

