# [MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction](https://arxiv.org/abs/2111.14973)

_December 2021_

tl;dr: Extension to multipath by taking in structured input.

#### Overall impression
Prediction need to fuse highly heterogeneous world state (static and dynamic) in the form of rich perception signals and map information, and infer highly multi-modal distribution over possible futures. 

This paper seems to be a [VectorNet](vectorNet.md) version of [Multipath](multipath.md). It on a high level is similar to [Multipath](multipath.md) in that the model consists of 1) an encoding step and 2) a predictor head which conditions on anchors and 3) outputs a Gaussian Mixture Model (GMM) distribution for the possible agent position at each future time step.

The paper also proposes multi-context gating (MCG) mechanism which is highly similar to cross attention. It also has a context vector which looks quite similar to what Andrej presented in Tesla AI day in their transformer architecture. 

The paper also has a great overview of past SOTA methods of behavior prediction.

#### Key ideas
- Sparse encoding of heterogeneous scene elements (instead of dense image-based encoding in [Multipath](multipath.md)).
- Multi-context gating (MCG) to fuse polylines (map) and raw agent state info.
- Learn latent anchor embeddings end to end in the model, instead of using static anchors. 

#### Technical details
- GMM (gaussian mixture model) is a 

#### Notes
- Questions and notes on how to improve/revise the current work
