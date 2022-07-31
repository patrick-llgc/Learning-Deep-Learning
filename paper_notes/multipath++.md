# [MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction](https://arxiv.org/abs/2111.14973)

_December 2021_

tl;dr: Extension to multipath by taking in structured (vectorized) input.

#### Overall impression
Prediction need to fuse highly heterogeneous world state (static and dynamic) in the form of rich perception signals and map information, and infer highly multi-modal distribution over possible futures. 

This paper uses sparse encoding of heterogeneous scene elements and is a [VectorNet](vectorNet.md) version of [Multipath](multipath.md). It on a high level is similar to [Multipath](multipath.md) in that the model consists of 1) an encoding step and 2) a predictor head which conditions on anchors and 3) outputs a Gaussian Mixture Model (GMM) distribution for the possible agent position at each future time step.

The paper also proposes multi-context gating (MCG) mechanism which is highly similar to cross attention. It also has a context vector which looks quite similar to what Andrej presented in Tesla AI day in their transformer architecture. 

The paper proposed to use separate encoders for each input modality. This is improved by [Wayformer](wayformer.md). 

**The paper also has a great overview of past SOTA methods of behavior prediction. A great starting point for behavior prediction.**

#### Key ideas
- Encoding: raster vs polyline
	- Raster encodes the world s a stack of images.
	- Polyline describes the world as piecewise linear segments. 
	- Drawback of raster vs polyline:Difficult in modeling long-range interactions, constrained by limited Receptive Field, and it is difficult to represent continuous physical state
- Input
	- road network encoding: closest P=128 polylines into their frames of reference. Road state vector includes 
		- closest point on segment, direction and magnitude
		- segment starting to end, direction and magnitude
		- vector pointing from start to closet point, magnitude
		- tangent vector of that piece
		- class of road segment, one-hot encoding.
	- agent state history: every other car, or ego-agent
	- agent interaction: permutation-invariant set operator: pooling or soft attention. AV-specific features are treated separately.
- Output trajectory: Mixed gaussian with anchors is the most popular representation
	- polynomials with smooth constraints
		- classification: 1/M modes
		- regression: M modes x T timestamp
	- underlying kinematics control signals
- **Multi-context gating (MCG)** 
	- as an alternative to cross-attention. It is used to fuse polylines (map) and raw agent state info.
	- MCG is an effective way to fuse multimodal input together. It uses a concept "context vector" to capture permutation-invariant context information. 
	- MCG can be stacked by having a skip connection with the original input.
- Learn latent anchor embeddings end to end in the model, instead of using static anchors. This is superior and inspired by [DETR](detr.md).

#### Technical details
- Cross attention is used to condition one input type (agent encoding) on another (road lanes).
- GMM (gaussian mixture model) is a practical way to handle mode collapse.
- Context vector of MCG is set to be all 1's when no input is specified.
- Trajectory can be obtained from acceleration using Verlet integration. It is a practical way to do integration for Newton equations.

#### Notes
- TNT/DenseTNT([github](https://github.com/Tsinghua-MARS-Lab/DenseTNT))
- [SceneTransformer](https://arxiv.org/abs/2106.08417): Waymo
