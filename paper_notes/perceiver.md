# [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)

_August 2021_

tl;dr: A general architecture to model arbitrary multimodal input. 

#### Overall impression
The paper proposes a general transformer architecture to model multimodal inputs such as image, video, audio, point cloud, etc.

[Transformer](transformer.md) has the quadratic scaling problem and thus cannot handle very large inputs. 

It still focuses on classification task, but it builds a foundation for other type of higher level tasks such as object detection, segmentation, etc.

#### Key ideas
- Computational cost reduction in transformers
	- Cross attention: Quadratic scaling to input size in transformers --> Linear 
	- Self-attention (latent transformers): independent of input size, and thus decouples depth of the network with input size
	- More efficient than efficient transformers (scales linearly with input size)
- Architecture: asymmetric attention mechanism to iteratively distill input into a tight latent bottleneck, allowing it to handle very large inputs. 


#### Technical details
- Transformers and input order invariant and thus need some guidance of positional encoding to disambiguate the order for some time of inputs. For other orderless type of inputs such as point cloud, no positional encoding is needed. 

#### Notes
- Questions and notes on how to improve/revise the current work  

