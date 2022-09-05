# [MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction](https://arxiv.org/abs/2208.14437)

_September 2022_

tl;dr: Much improved version of [VectorMapNet](vectormapnet.md) by introducing permutation-invariant loss.

#### Overall impression
[VectorMapNet](vectormapnet.md) proposes to use **polyline primitives** (to be exact, *directed polylines*) to model map elements, and MapTR takes this idea further by also introducing permutation-invariant loss in the training.

Two different ways to handle set, which is inherently orderless. Either to process it with symmetric operation, such as maxpool or average-pool or min/max as in [PointNet](pointnet.md), or use min/max in the loss formulation to iterate through all permutations. --> The latter idea seems to have been popularized by [monodepth2](monodepth2.md) when people realize they could plug in min/max operator in the loss formulation while keeping differentiability.

Instead of the coarse to fine two-stage method in [VectorMapNet](vectormapnet.md), MapTR uses instance queries and point queries and concatenates them to indicate a particular point from a particular instance. --> Maybe point queries is unnecessary, when polyline is viewed as a more complicated bbox.

#### Key ideas
- Major issue with representing map elements as **directed polylines** is that this assumes all map elements have explicit definition of start point and direction. Yet this is not true as there exists multiple equivalent polyline formulation and this can cause confusion for the network training.
- GT: y = (c, V, Γ), y-hat = (p-hat, V-hat)
- Matching cost:
	- Instance level matching: L(p, c) + L_position(V, V-hat)
	- Point level matching: D_manhattan(V, V)
- End to end training with 3 losses
	- Classification loss: focal loss (improved CE)
	- point2point loss: D_manhattan(V, V)
	- Edge direction loss: make sure edge directions are the same between pred and GT. --> This is 2nd order effect, but looks like quite effective in the ablation study. This is almost like the PAF supervision.


#### Technical details
- A polyline with N points has 2 equivalent permutations, and a closed polygon with N points has 2N equivalent permutations. --> Note that this is not total permutation with N!.
- Front/rear 30 m, left/right 15 m. 
- Each map elements are represented with 10 points. --> This is ablated but also seems one arbitrary design.
- BEV transformation via [GKT](gkt.md), which produces a dense BEV feature map. The performance is about the same as [BEVFormer](bevformer.md) as shown in the ablation study.
- **Decoder layer number matters!** Huge improvement from 1 to 2, and incremental improvement above 2. Maybe one layer of decoder is indeed not enough. 

#### Notes
- To use lane or not, this is a critical question for planning to answer.
- 
