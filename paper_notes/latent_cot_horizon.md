# [A Survey on Latent Reasoning](https://arxiv.org/abs/2507.06203)

_January 2026_

tl;dr: Summarize the the main idea of the paper with one sentence.

#### Overall impression
Reasoning is fundamentally a continuous, iterative computation—and language-based CoT is just a lossy interface, not the reasoning itself.


#### Key ideas
- Explicit CoT is a bandwidth bottleneck. Roughly more than 3 orders of magnitude of information bottleneck boost. 
	- Language tokens carry ~10–20 bits per step (Log2(vocab_size), log2(1e6) = 20 bits for a million vocab size). 
	- In contrast, the internal "latent" vectors (the hidden states) are usually high-dimensional (e.g., 4096 dimensions). If each dimension is a 16-bit float, a single latent step could theoretically carry 65,536 bits of information.
- Two main latent reasoning paradigms
	- Vertical (depth) recurrence: reuse layers repeatedly to “think longer” (looped / recurrent depth).
	- Horizontal (state) recurrence: evolve a hidden state over time (RNN/SSM/optimizer-style memory).
- Explicit CoT can be compressed into latent space
	- Training can internalize step-by-step reasoning into activations, achieving similar accuracy with much lower latency and token cost. See pioneering work such as [Stepwise Internalization](stepwise_internalization.md) and [Coconut](coconut.md).

#### Technical details
- <!--Summary of technical details, such as important training details, or bugs of previous benchmarks.-->

#### Notes
- <!--Questions and notes on how to improve the current work-->

