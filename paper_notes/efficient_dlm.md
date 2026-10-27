# [Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed](https://arxiv.org/abs/2512.14067)

_January 2026_

tl;dr: AR pretrained LLMs can be efficiently deployed to onboard device via diffusion.

#### Overall impression
Effective AR-to-dLM conversion only requires 10% of the original token in CPT. The language knowledge, syntax, reasoning skills already exist in the AR weights.

Essentially we pay 10% more training cost and get 2-4 times 

One highlight: we can dynamically adjust TPF (token per forward in runtime policy decision, not a model decision). This is controlled direclty by a conf thresh.


#### Key ideas
- Difference with specdec
	- SpecDec: Two models, Binary accept/reject
	- Efficient-DLM: One model, Continuous speed–accuracy curve
- confidence 
	- context: context means all tokens that are currently not masked at this iteration
	- “Given what I currently know, how peaked is my belief about this token?”
	- pi​(v)=p(xi​=v∣current sequence)
- conf-based decoding 
	- based on conf > thresh, instead of top-K or fixed schedule
	- Dream and llada uses fixed decoding schedule
- key concepts
	- NFE (Number of Function Evaluations)
	- TPF (Tokens Per Forward) = N / NFE

	
#### Technical details
- <!--Summary of technical details, such as important training details, or bugs of previous benchmarks.-->

#### Notes
- <!--Questions and notes on how to improve the current work-->

