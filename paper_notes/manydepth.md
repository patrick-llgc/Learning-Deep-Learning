# [Manydepth: The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth](https://arxiv.org/abs/2104.14540)

_January 2022_

tl;dr: Efficient and accurate use of multiframe information for monodepth.

#### Overall impression
For monodepth application, sequence information is often available at test time. There exists two ways to leverage multiframes for monodepth estimation. First uses expensive test-time refinement techniques ([Consistent Video Depth](cvd.md), [Robust CVD](robust_cvd.md), CoMoDa, SSIA, etc) or recurrent network (). ManyDepth is an adaptive approach to dense depth estimation that can make use of seq info at test time when it is available.

The paper provides a good overview of recent advances of self-supervised monodepth.

>> ManyDepth address what was thought to be a forced choice in 3D reconstruction, between classic triangulation over multiple frames versus instant-but-fragile single-frame inference with a neural network. ([Source](https://nianticlabs.com/blog/manydepth-research/?hl=en))


#### Key ideas
- Formulation
	- Train with many frames (both previous and future)
	- It does not rely on any semantic model
	- Can take in one or multiple frames during test time.
- Builds on two well established components
	- Self-supervised reprojection based training (after [SfM-Learner](sfm_learner.md), [Monodepth2](monodepth2.md))
		- Training with both t+1 and t-1 frames
		- Test with only current t frame
	- Multi-view cost volume 
		- Set a series of depth plane $d \in P$, and set d_max and d_min
		- Source image t-1 is encoded into a feature F_t-1 (with dim H/4 x W/4 x C), warped into time t with hypothesized depth d and estimated pose from PoseNet. 
		- For each d, L1 loss between warped feature and the feature F_t (with dim H/4 x W/4 x C), we get one slice of cost volume (with dim H/4 x W/4 x 1).
		- Build entire cost volume (with dim H/4 x W/4 x |P|).
		- Concat the cost volume with Ft into depth prediction encoder-decoder. (In monodepth, Ft only without cost volume)
- Advantages/Innovations
	- Adaptive Cost Volumes
		- d_min and d_max are learned from training data, with the use of a running average. The running average is then used in test time. --> Like BN.
	- Moving objects: Use single view monodepth to supervise. Discard this network at test time.
		- Naively concatenating cost volume to the feature Ft leads to bad test time performance (overfitting)
		- **Cost volume only works well in textured region in a static setting.**
		- network may become over-reliant on the cost volume, and inherit the cost volume's mistakes
		- Use a single image depth network to regularize, but only in regions (within the "motion mask") where there is a large gap between multi- and single-image predictions.
	- Static scenes and start of sequence: Data augmentation by feeding identical images. --> This is similar to the training of [CenterTrack](centertrack.md).
		- Static scenes: during training, replacing the cost volume with a tensor of zeroes with probability of p (=0.25).
		- Start of sequence: replace I_t-1 input with It with probability of q (=0.25).
- Architecture
	- $\theta_{pose}$: pose estimation
	- $\theta_{consistency}$: single-frame depth, disposable 
	- $\theta_{depth}$: multi-frame depth
	
	
#### Technical details
- TTR (test time refinement) 
	- Multiple forward and backward passes is needed for each set. Several seconds to complete. --> This may be useful for offline perception.
	- One pass may be already effective and efficient ([CoMoDA](comoda.md)).
- Recurrent approaches
	- Much more efficient compared to test-time-refinement. 
	- Cons: They do not explicitly reason about geometry during inference. Inaccurate.
- MVS (multiview stereo)
	- unordered image collection
	- ManyDepth is superior to MVS in that most MVS methods assume there is no moving objects in the scenes, and assumes that the camera is not static. 


#### Notes
- [Code on Github](https://github.com/nianticlabs/manydepth)
