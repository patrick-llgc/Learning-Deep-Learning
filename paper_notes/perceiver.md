# [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)

_August 2021_

tl;dr: A general architecture to model arbitrary multimodal input. 

#### Overall impression
The paper proposes a general transformer architecture to model multimodal inputs such as image, video, audio, point cloud, etc. Transformers have been rapidly percolating into perception.

[Transformer](transformer.md) has the quadratic scaling problem and thus cannot handle very large inputs.

It still focuses on classification task, but it builds a foundation for other type of higher level tasks such as object detection, segmentation, etc.

Efficient implementation of transformers include [Set Transformer](set_transformer.md) and [Linformer](linformer.md). See [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) for a review. Perceiver is more scalable than linear as it decouples the computation from length of the input.

- Set transformer uses cross attention to project a large input array to a smaller array, either to reduce the computation within a module or to project inputs to a target output shape. 
- Perceiver uses cross-attention over an auxilliary low-dimensional array to reduce the complexity of attention from quadratic to linear in the input size.
- Linformer produces linear complexity self-attention modules by projecting key and value inputs to arrays with a size smaller than the input.

#### Key ideas
- Computational cost reduction in transformers
	- Cross attention: Quadratic scaling to input size in transformers --> Linear. Cross attention maps to a fixed-dimension latent variable and also decouples the network depth from the input size.
	- Self-attention (latent transformers): independent of input size, and thus decouples depth of the network with input size
	- More efficient than efficient transformers (scales linearly with input size)
- Architecture: asymmetric attention mechanism to iteratively distill input into a tight latent bottleneck, allowing it to handle very large inputs. 
	- The weights between the transformer towers can be optionally shared, and thus the model can be interpreted as RNN, unrolled in depth using the same input, rather than time.

![](https://miro.medium.com/max/1400/1*41GYOpmCItZMxO4V7U4FGw.jpeg)

#### Technical details
- Transformers and input order invariant and thus need some guidance of positional encoding to disambiguate the order for some time of inputs. For other orderless type of inputs such as point cloud, no positional encoding is needed. 
	- We can compensate for the lack of explicit structures in our architecture by associating position and modality specific features with every input element. These can be learned or constructed using high fidelity Fourier features. 
- Original CNN literature: (Fukushima 1980, LeCun 1998, Ciresan 2011, Krizhevskey 2012)
- QKV attention applies three networks, query, key and value networks which are typically MLP to each element of the input array, producing 3 arrays with the same length. --> [Transformers](transformer.md).
- Positional encoding is concatenated, rather than added to the input like the original transformers.
- LAMB optimizer for transformer models.
- Ablation studies
	- Learned positional encoding leads to performance drop.
	- Permuted image input will not affect trasnformers and perceiver model but affects [ViT](vit.md) and ResNet models, even though the input are the same with positional embedding.
- Audio inputs: raw signal vs mel spectrogram (mel(ody)-scale). Most previous methods use spectrogram but Perceiver can do just as well with raw input. --> Perhaps this can be used in 

#### Notes
- Questions and notes on how to improve/revise the current work  

