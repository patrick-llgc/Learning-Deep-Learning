# [Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)

_January 2026_

tl;dr: Reason in unconstraint latent space vs language space to bypass the "Vocabulary Bottleneck".

#### Overall impression
Coconut allows more efficient reasoning, reducing token num and also induces new thining patterns.

The latent thinking requires a normal training with explicit CoT first. It creates a latent thinking stage marked by <bot> and <eot>, and iteratively repalces each step of explicit thinking (1 thinking step ≈ a short sentence or clause, x language tokens) with c latent thought tokens. 

One possible drawback is that this cannot decode into human language and is hard to visualize. Maybe [DLCM](dlcm.md) is a better method. 


#### Key ideas
- More efficient reasoning
	- Replaces each reasoning step with c continuous tokens
	- How does this reduce inference steps? c is a hyperparameter that can be finetuned. 
- Induces BFS
	- Standard LLMs are DFS as every time a word is generated, the model's internal probability distribution collapses into a single token. And LLM cannot do backtrack easily (only explcitly with CoT at grammatically correct tuning point).
	- Entropy analysis: in early latent steps, the model assigns high and nearly equal probabilities to multiple different valid next steps.
	- Latent Coconut recovers.

#### Technical details
- <!--Summary of technical details, such as important training details, or bugs of previous benchmarks.-->

#### Notes
- The iterative nature of latent token is very similar to diffusio language models. The connection is also highlighted in the [latent CoT survey](latent_cot_horizon.md).
	- In both diffusion and latent CoT, "commitment" is delayed until the end. Commitment ≈ decoding into natural language. (h  →  softmax(W·h)  →  token)