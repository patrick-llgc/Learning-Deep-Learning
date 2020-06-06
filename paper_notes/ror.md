# [RoR: Rules of the Road: Predicting Driving Behavior with a Convolutional Model of Semantic Interactions](https://arxiv.org/abs/1906.08945)

_April 2020_

tl;dr: Multi-modal **behavior prediction** with perception output.

#### Overall impression
[ChauffeurNet](chauffeurnet.md), [IntentNet](intentnet.md) and [Rules of the Road](ror.md) all uses semantic map that includes static and dynamic information. 

For behavior prediction, we need past history of the agent, dynamics of other entities and semantic scene info. **The paper does not specifically tackle the problem of motion planning.** Rather it still focuses on the prediction of a target entity. This target entity could be ego car or other cars. does not explicitly perform prediction of other agents. 

This is to be differed from the planning-centric [ChauffeurNet](chauffeurnet.md). For motion planning, collision should be explicitly modeled. 

The main novelty of this paper seems to be in the semantic map encoding into a 20-channel pseudo-map. 

The rest of the paper is not well written. Lots of details are left out and the main body of the paper only have 8 pages for CVPR. 

#### Key ideas
- Input representation:
	- Semantic map in RGB image with 3 channels	- Entity representation: ego entity has 7 channels [CenterNet](centernet.md)-like, one heatmap for (x, y), and v, a and covariance matrix norms. Other entities are also encoded in another 7 channels. Only the center of the bboxes have values. 
	- Dynamic contexts: 3-ch RGB image with all oriented bboxes colored by the class. Also contains traffic light by masking out road connections. 
	- This representation can be easily augmented with additional information, such as TSR or pedestrian guestures. 
- Multi-model output. To address exchangeability and collapse of modes, a latent variable z is introduced. --> Why can latent variable help? This exchaneability issue can be solved by using a fixed set of anchors, as in [MultiPath](multipath.md).

#### Technical details
- Normalize by 99th percentile to avoid outliers. 

#### Notes
- Industry favors decoupled pipelines. This gives better modularity and leads to better engineering management. Academia seems to like end-to-end system. This seems to decouple the publications from Waymo, Zoox or especially from Uber ATG.
