# [Wayformer: Motion Forecasting via Simple & Efficient Attention Networks](https://arxiv.org/abs/2207.05844)

_July 2022_

tl;dr: Simplistic neural network architecture to handle multi-modality input for prediction network.

#### Overall impression
Wayformer is a simple and homogeneous network architecture to handle diverse and heterogeneous inputs to the motion forecasting task.

This paper seems to be heavily inspired by [Perceiver](perceiver.md) and [Perceiver IO](perceiver_io.md). Both Perceiver and Perceiver IO are pioneering work on unifying multimodal input. [Wayformer](wayformer.md) views the input to prediction (aka motion forecasting) networks as multi-modality (road geoemtry, lane connectivity, time-varying traffic light state, and history of a dynamic set of agents and their interactions).

Many previous work ([Multipath](multipath.md), [Multipath++](multipath++.md)) focus on handling multimodality of the **output** space of prediction network, while [Wayformer](wayformer.md) focuses on the **input** space. [Wayformer](wayformer.md) also builds on top of [Multipath++](multipath++.md) and uses its decoder and loss design. This suggests that motion forecasting also **conforms to Occam's Razor.** 

The paper also only predicts the future for a single agent. This is very different from [Scene Transformer](scene_transformer.md) which predicts the entire scene at once (similar to the difference between single stage and two stage methods for object detection).

There are many facets of motion forecasting (behavior prediction). This can be compared with the review session of [Multipath++](multipath++.md).

- Input representation: rasterized (multipath) vs structured (multipath++)
- Fusion strategy: late fusion (multipath++) vs early fusion (wayformer)
- Fusion module: GNN, soft attention, Transformer (and more efficient variants)
- Decoding module: pooling, multimodality, target driven, etc
- Decoding scheme: entity-centric (wayformer) or scene-centric (scene transformer)


#### Key ideas
- There are two distinctive concepts both denoted as multi-modality in motion forecasting. This also makes modeling for scene understanding challenging. --> Both can be translated as 多模态 in Chinese, which is also unfortunately inconvenient.
	- Input multimodality: the input to motion forecasting network is **heterogeneous**, such as road geometry, lane connectivity, time-varying traffic light state, and history of a dynamic set of agents and their interactions. 
	- Output multimodality: the output of motion forecasting need to be **diverse** to reflect many possible underlying intents of the traffic agents, such as going straight, turning right, or making u-turns. 
- Overall structure
	- The scene encoder is one or more attention encoders that summarize the driving space.
	- The decoder is a stack of standard transformer cross attention blocks, learned initial queries are fed in, and then cross-attended with the scene encoding to produce trajectories.
- Two common techniques to speed up self-attention: (original self-attention is multi-axis attention)
	- Factorized attention: applying self-attention over each dimension. The factorized attention blocks can be arranged in Sequential Attention or Interleaved Attention.
	- Latent query attention --> from [Perceiver](perceiver.md) and [Perceiver IO](perceiver_io.md).
		- Use latent queries to control the feature dimension, so that the architecture depth (computation needed) can be decoupled from input length (or resolution)
- Ablation studies
	- Fusion scheme vs capacity
		- Late fusion is computationally cheap as there is no interactions between modalities during scene encoding.
		- With moderate model capacity, there is no obvious performance difference.
	- Factorized attention: helps boost performance with early and late fusion.
	- Latent query attention: no performance loss with x16 compression.

#### Technical details
- Projection layer: like 1x1 conv, to unify feature dimension.
- Fusion can be summarized as $Z = Encoder({m_0, m_1, ..., m_k})$, where $m_i \in {R}^{A \times (T \times S_m) x D}$, $Z \in {R}^{A \times L \times D}$
- [Wayformer](wayformer.md) follows the trajectory decoding in [Multipath++](multipath++.md), using a combination of classification and regression losses. 
- Trajectory aggregation are like the NMS in object detection.
- Waymo is 1+8@5Hz, Argoverse is 2+3@3Hz.

#### Notes
- Page 3: there is a typo (or error in concept). Positional embedding is needed to handle the **permutation invariance** instead of **permutation equivarience**. PE + self attention is permutation equivarience. 
