# [TNT: Target-driveN Trajectory Prediction](https://arxiv.org/abs/2008.08294)

_February 2022_

tl;dr: Predict diverse set of future targets and then use target to drive trajectory prediction.

#### Overall impression
The paper described the core drawbacks of previous methods, involving sampling latent states (VAE, GAN), or fixed anchors ([coverNet](covernet.md), [MultiPath](multipath.md)). 

TNT has the following advantages

- supervised training
- deterministic inference
- interpretable
- adaptive anchors
- likelihood estimation

The target, or final state capture most uncertainty of a trajectory. TNT decompose the distribution of futures by conditioning on targets, and then marginalizing over them.

The anchor-based method is improved by [DenseTNT](dense_tnt.md) to be anchor-free, which also eliminated the NMS process by learning.

#### Key ideas
- Step 1: target prediction, based on manually chosen anchors
- Step 2: Motion estimation, conditioned on targets
- Step 3: Trajectory scoring/selection, with scoring and NMS

#### Technical details
- Vectorized (sparse) encoding with [VectorNet](vectornet.md).

#### Notes
- [CoRL talk on Youtube](https://www.youtube.com/watch?v=iaaCbKncY-8)
