# [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)

_August 2021_

tl;dr: Perceiver that generalizes beyond classification task to dense prediction and multimodal output tasks.

#### Overall impression
The general architecture follows that of [Perceiver](perceiver.md), but the output is more versatile and thus [Perceiver IO](perceiver_io.md) can handle more tasks than classification.

![](https://miro.medium.com/max/1400/0*0QENq9YFSQcaWvwy.png)

The paper seems to be heavily inspired by [DETR](detr.md) by adding a output query to handle the dense prediction tasks.

Both [Perceiver](perceiver.md) and [Perceiver IO](perceiver_io.md) are "fully attentional network" (FAN?). They use **read-process-write** architecture: inputs are read (encoded) into a latent space, processed by transformers and then written (decoded) to produce outputs. 

#### Key ideas
- Architecture: maps arbitrary input arrays MxC to arbitrary output arrays OxE, in a domain agnostic process.
	- The **key insight of Perceiver IO** is that we can predict each element of the output using another attention modules, by simply **querying the latent array** with a query feature vector unique to the desired output elements. The output has the same number of elements as the output query. 
	- The bulk of computation happens in latent space whose size is smaller than input and output, making computation tractable for very large inputs and outputs. 
	- The latent array to query the input NxD (can be chosen to make computation tractable) and output query O can be hand-designed, learned embeddings, or a simple function of the input.
- Output query
	- For classification, a learned feature can be reused for all examples
	- For tasks with spatial/sequence structure, use positional encoding
	- For multitask/multimodal, a single query for each task
	- Input features can also be concatenated to enhance the task (like optical flow)
- Dense task: Optical flow
	- Input: every pixel has 3x3x3x2 = 54 values as features. A fixed position encoding to the features. 
	- Output query: using same query representation using the same encoding used for the input (feature + positional embeddings).

#### Technical details
- Optical flow: temporal image gradient is also important (per Lucas-Kanade algorithm). Ignoring the lighting effects, the temporal gradient of the image is equal to the spatial gradient times the spatial velocity.
- Optical flow: first find correspondence between points, then it must compute the relative offset of these points, then it must propagate the estimated flow across large region of space, even for areas without texture. 

#### Notes
- Questions and notes on how to improve/revise the current work  

