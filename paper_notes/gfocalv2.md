# [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://arxiv.org/abs/2011.12885)

_December 2020_

tl;dr: Introduces a bridge between QFL and DFL in [generalized focal loss](gfocal.md).

#### Overall impression
The improvement of GFocalV2 over [GFocal](gfocal.md) is simple. [GFocal](gfocal.md) predicts QFL (quality focal loss) and DFL (distribution focal loss) separately, while [GFocalV2](gfocalv2.md) introduced a bridge named distribution guided quality predictor (DGQP) to guide the prediction of localization quality estimation (LQE). 

The classification and localization still has a joint representation, but the training and prediction method has a decomposed design.

The acronym in this manuscript is getting a bit crazy. Even in the original [GFocal](gfocal.md) paper, the annotation is a bit unnecessarily obscure.

#### Key ideas
- Use the statistics of bbox distribution, instead of vanilla convolutional features. 
	- Uses topK values (k=4) and their mean value as features
	- A light subnetwork with two FC layers predicts the IoU
	- The joint representation is the classification times the IoU. 
- Generalized focal loss v2 leads to ~2 AP increase in all one-stage object detectors, without sacrifice on training or inference speed. (无痛涨点)


#### Technical details
- Using regression features to enrich the classification features to predict the joint cls-IoU representation
- The added bridge does help ease the training of the QFL, as evidenced by lower loss on IoU score prediction.

#### Notes
- GFocalV2 still uses distribution integral to predict bbox edge locations. Why not use topK from the distribution instead to address potential multi-modal problem?
- The results of this paper is not that convincing. If the distribution shape helps with the prediction of QFL, wouldn't the use of only the top1 value help QFL? In the ablation study this is not the case.

