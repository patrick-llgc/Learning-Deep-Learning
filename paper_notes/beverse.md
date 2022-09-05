# [BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving](https://arxiv.org/abs/2205.09743)

_June 2022_

tl;dr: One-stage, multi-task BEV perception and prediction.

#### Overall impression
This paper combines the recent progress in static BEV perception (such as [HDMapNet](hdmapnet.md)), dynamic BEV perception ([BEVDet](bevdet.md), [BEVFormer](bevformer.md)) and motion prediction ([FIERY](fiery.md)). The motion prediction part largely inherits the spirits of [FIERY](fiery.md).

This paper claims to be the 1st paper that performs joint perception and motion prediction, but actually [FIERY](fiery.md) should be. [BEVerse](beverse.md) also added static perception to FIERY. The joint perception and prediction idea has also been exploited in lidar perception field, such as [FAF](faf.md).

The paper's major contribution seems to be the iterative flow for efficient future prediction. Yet the recurrent roll-out method of future prediction is unfriendly to realtime performance in production. Transformer-based method which can predict multimodal future waypoints all at once may be the way to go.

Although BEVerse achieves highest NDS score, this mainly comes from the improved velocity estimation (reduced mAVE error). The mAP is actually worse than most BEV detection work (BEVDet, PETR). [BEVDet4D](bevdet4d.md) achieves better performance in object detection in both mAP and NDS.

This paper reminds me of [BEVDet](bevdet.md), which exhibits great engineering skills with great results from existing technical components, but the writing of the manuscript leaves much to be improved. 


#### Key ideas
- The drawback of sequential engineering design
	- propagation of errors can significantly influence the downstream tasks.
	- Extra computational burden due to repeated feature extraction.
- The Spatiotemporal alignment module in BEV encoder largely follows that of [FIERY](fiery.md). It transforms past BEV features to the present coordination system with ego motion and processes the aligned 4D features with the spatiotemporal BEV encoder.
- The data augmentation strategy is from [BEVDet](bevdet.md).
- Grid sampler: crop task specific regions and transform to ideal resolution through by-linear interpolation. The learning of semantic map requires fine-grained features because the lane lines are narrow in 3D space.
- Head for motion prediction has two improvements from FIERY
	- upgrade sampled global latent vector which are shared for each BEV pixel to a latent matrix
	- future states are initialized from sampled latent vector **and a predicted flow (offset)**


#### Technical details
- BEV perception vs mono3D: mono3D methods separately process the image of each view and merge the outputs with heuristic postprocessing.
- Prediction time window (2+1 --> 4) is the same as FIERY.


#### Notes
- Link on [github](https://github.com/zhangyp15/BEVerse) (to be released as of 2022/06)
