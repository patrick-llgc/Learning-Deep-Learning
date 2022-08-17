# [HOME: Heatmap Output for future Motion Estimation](https://arxiv.org/abs/2105.10968)

_August 2022_

tl;dr: Using heatmap as output is a flexible way to balance MR and ADE. 

#### Overall impression
The paper itself did not propose super innovative ideas. Instead it demonstrates a solid engineering work, and provides more insights into the advantages of the existing technical components.

The paper has two major contributions. First it uses CNN + attention in the backbone in parallel to encode heterogeneous input. Second it shows that the heatmap output (following the example of [TNT](tnt.md)) is a very flexible way to  represent output as it allows different sampling strategy to tune the performance without retrain the model.

> The challenge in behavior prediction is to avoid big failures where a possibility (a certain mode) is not considered at all. This has more impact to the safety of autonomous driving than having the absolute closes trajectory to the ground truth. --> This is the first principle we need to bear in mind when doing projects in behavior prediction. Multimodal is your friend, and mode collapse is your enemy.

In a way, [HOME](home.md) is a dense form of [TNT](tnt.md) and thus more like [DenseTNT](densetnt.md).

#### Key ideas
- Ways to avoid mode collapse
	- Anchor based: anchor trajectory ([Multipath](multipath.md)) or anchor target ([TNT](tnt.md))
	- heatmap based ([HOME](home.md))
- Input representation 
	- Context(total of 45 ch)
		- HD map as 5-ch raster image: drivable area, lane boundaries and directed centerlines with headings are encoded with hsv
		- target agent history: 20 ch, as a moving square
		- other agents history: 20 ch
	- Scalar history: (H, 4). 3 for location and 1 for a binary padding mask.
	- Agent history: (N, H, 4). Similar to scalar history for the target agent, for each of other agent.
	- The input representation is fed into a CNN to generate a **context feature map**.
- Inter-agent attention for interaction
	- Cross attention to build a 128-d feature vector to summarize the interaction between the target agent and other agents. Then it is tiled to be concatenated with the context feature map.
- The training of the heatmap is similar to that of [CenterNet](centernet.md).
- Sampling strategies
	- MR (miss rate)-centric: maximize coverage. A case is defined as missed if the ground-truth is further than 2 m from the prediction. **Sample greedily** in the heatmap, and remove area within 2 m from the sampled point, then sample the highest probability area again. And repeat. 
	- FDE-centric: inspired by kMeans, and improved by soft weighting. This sampling strategy targets to guess the final destination as accurately as possible, even at the risk of missing out some mode sometimes. The initial seed of kMeans is provided by MR-centric. Controlling the iteration number will control the trade-off between MR and FDE.
- Generation of full trajectory with MLP --> How the separate model of full trajectory generation is trained is not detailed here. Maybe like that of [TNT](tnt.md), where the loss is Huber loss over per-step offset.

#### Technical details
- Ablation study on heatmap format: it is much better than regression based output representation. The study also showed that this advantage did not come from the preservation of spatial information. Even if losing the spatial context by adding one global pooling bottleneck layer, the performance does not drop much.
- Regression based output does not allow finetuning postprocessing to change the behavior of the entire system. Also, if a model is trained with k=6 modes, and output changes to 2, this could lead to performance degradation in MR, as the output modes are not necessarily sorted to optimize MR. Therefore **a regression based model needs to be retrained**. Heatmap based model only needs to change the sampling strategy.

#### Notes
- Questions and notes on how to improve/revise the current work
