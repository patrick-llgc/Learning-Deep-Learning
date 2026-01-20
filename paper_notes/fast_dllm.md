# [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)

_January 2026_

tl;dr: Block decoding, by enabling kV cache for diffusion LLM, and confidence aware parallel decoding. 

#### Overall impression
Approximate the KV-cache by noticing that neighboring decoding timesteps have very similar KV attention map.

#### Key ideas
- Exploits high similarity between iteration steps to reuse KV values (KV cache), and update KV cache periodically.
- Confidence level based parallel decoding
	- Why does it solve inconsistencies (in grammar/logic) by parallel decoding?
	- Confidence-based decoding only unmask a token when its confidence score (e.g., probability, embedding consistency) is high, which signals that its **contextual dependencies** are already resolved. 
	- Low-confidence tokens continue to receive **contextual updates** from already unmasked tokens.
	- Safety valve for progress: unmasking the single highest-confidence token when no others meet the threshold.
- threashold vs factor based decoding
	- Threshold decoding is slower but more accurate

#### Technical details
- Block size by fast-dLLM is 32.
- Why KV-cache cannot be used naively for DLMs?
	- DLMs generate the full sequence via denoising steps, not token-by-token left-to-right
	- KV pair values are not static/reusable across denoising steps (entire sequence is refined each time)
- Fast-dLLM's fix
	- Split sequences into fixed-size blocks, then reuse block-level KV pairs across adjacent denoising steps (~1 cosine similarity between step-wise KVs justifies reuse). 
	- This also include DualCache to extend reuse to masked suffix tokens (even though they are not demasked yet)
	- KV cache refresh/update
		- KV cache needs to have "refresh" periodically to prevent drift
		- Reuses block-level KVs across adjacent steps
		- Updates the cache of all blocks after each block’s decoding. The update is lightweight and folded into decoding, with no extra overhead.
- Q: Does Fast-dLLM’s confidence-based decoding guarantee blocks are decoded in a fixed number of steps
	- Confidence-based decoding is dynamic, tokens unmasked per step varies per step but generally exceed 1 
	- unmask top-confidence token if no tokens meet threshold to prevent an infinite loop

#### Notes
- Unmasking schedule: S-shaped saturating curve
	- t=1 to t≈T/3: Slow progress. Few tokens (high-confidence anchors) unmask per step
	- t≈T/3 to t≈2T/3: Fast progress. Many tokens unmask per step (stable context enables batch unmasking)
	- t≈2T/3 to t=T: Few remaining tokens unmask per step (hard dependencies take time)

