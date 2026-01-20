# [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/abs/2509.26328)

_January 2026_

tl;dr: Adapt AR into a [block diffusion](block_diffusion.md) model (hybrid model, blockwise AR + intra-block diffusion), and inference intra-block with [Fast-dLLM](fast_dllm.md).

#### Overall impression
The combinatio of [block diffusion](https://arxiv.org/abs/2503.09573) + [fast-dLLM v1](fast_dllm.md). 

Both v1 and v2 enforce left-to-right blockwise AR dependency. But v2 adapts the training of the model and aligns it better with inference (训推一致).

Note that Fast-dLLM v2 is a hybrid model (block diffusion model) and is NOT a Full Diffusion Model.

#### Key ideas
- V2 is a lightweight adaptation of an AR to a intra-block diffusion model.
	- v2 does not modify the pre-trained AR LLM’s architectur
- Specialized attention mask of 2Lx2L size. 
	- Noised sequence is concatenated with clean sequence and thus the 2L length
	- Vectorized training via special pattern.
![](https://pica.zhimg.com/v2-f1ecce08791e8f5bc405be81d917c2e8_1440w.jpg)

#### Technical details
- Whether to refresh for decoded blocks in infernece
	- v1 refresh the KV cache of decoded block
	- v2 inference does NOT update the kV cash after a block is decoded, as the model is decoding blockwise in an AR way
- V2 inference
	- Inter-block (from block to block), use standard KV cache.
	- Intra-block (within a block), V2 also uses the inference techniques for V1. V2 devides each block into subblocks, and the dualcache happens on the subblock level (v2 subblock == v1 block, smallest unit of diffusion-style unmasking).

#### Notes
- DualCache for sub-blocks does NOT include the suffix cache for undecoded blocks beyond the current block being decoded.
	- Inter-block: Strict left-to-right AR dependency (no access to future blocks).
	- Intra-block: Bidirectional attention (access to all tokens within the current block, including masked/suffix tokens in the same block).

