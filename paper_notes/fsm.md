# [FSM: Full Surround Monodepth from Multiple Cameras](https://arxiv.org/abs/2104.00152)

_April 2021_

tl;dr: 360 deg monodepth prediction from self-supervision.

#### Overall impression
This paper pushes the application and the frontier of monodepth. It extends the scope of the previous works on monodepth from single camera to a multi-camera setting. 

FSM is intended for multi-camera config with very large baselines (and thus minimal image overlap). Stereo rectification and disparity estimation is not feasible.

Along the line of view synthesis, FSM introduces two new constraints for optimization and one benefit: STC (spatio-temporal constraint) and PCC (pose consistency constraints), and scale-awareness introduced by metric scale extrinsics. 

The [FSM](fsm.md) assumes known camera extrinsics (more precisely, it only requires relative extrinsics between cameras).

#### Key ideas
- STC: the spatiotemporal includes the reprojection constraints based on not only the estimated ego motion, but also a fixed spatial transformation between cameras.
	- Theoretically, the estimated ego motion does not have metric scale, but with joint training with the metric scale extrinsics, the model can be scale aware. See [PackNet](packnet.md), [SC-Sfmlearner](sc_sfm_learner.md) and [TrianFLow](trianflow.md) for other scale awareness solution.
	- The results trained with known extrinsics between cameras enables generation of scale aware models, with minimal degradation from the median-scaled counterpart. 
	- In a way, it uses the overlap region as stereo supervision to train the depthNet. It can improve individual camera performance.
- PCC: the pose consistency constraints dictates that the ego pose estimated by different cameras should be the same across multiple cameras, if the pose is transformed into a canonical coordinate system. In other words, **FSM enforces the learning of a smilier rigid motion for all cameras**. The transformation would require the relative extrinsics from one camera to the other.
	- The loss term encourages the R (3DoF) and t (quaternion) in the predicted pose are consistent, similar to the formulation in [PoseNet](posenet.md).
- Masking
	- Self-occlusion mask (due to camera positioning that resulting the platform partially covering the image). 
	- Non-overlapping mask: FSM only leverages the overlapping region during the loss calculation of the spatial constraints. --> Do they use non-overlapping mask in temporal component as well? Not stated explicitly and not likely.
	- Auto-masking: introduced by [monodepth2](monodepth2.md) to address the infinite depth issue. 

#### Technical details
- The DDAD (dense depth for autonomous driving) dataset is quite similar to what a mass production autonomous driving dataset would look like.
- Traditional approach to 360 perception may involve omnidirectional or panoramic images. 
- **Median scaling** is usually used in the evaluation of depth estimation at test time. In a multicam setup, we can enforce a single scaling factor instead of a factor for each camera. 
- The overlap between nuScenes is even smaller than DDAD, making it quite challenging.

#### Notes
- The codebase is based on [Monodepth2](monodepth2.md) and will be released soon.
- During training multiple camera images are used at the same time but the feature map are not combined in anyway, and multiple inference path is only connected by the loss function. The network still takes in one image at a time in inference time. --> Maybe next time we should do some sort of feature pooling to couple the inference pipeline more. 
- In future, the authors of FSM said they will relax the constraint of known extrinsics to enable **self-calibration**. This is along the lines of how [LearnK](learnk.md) extends [SfM-Learner](sfm_learner.md).