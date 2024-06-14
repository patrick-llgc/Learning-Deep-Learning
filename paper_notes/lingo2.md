# [LINGO-2: Driving with Natural Language](https://wayve.ai/thinking/lingo-2-driving-with-language/)

_June 2024_

tl;dr: First closed-loop world model that can output action for autonomous driving.

#### Overall impression
This is perhaps the second world-model driven autonomous drving system deployed in real world, other than FSDv12. Another example is [ApolloFM (from AIR Tsinghua, blog in Chinese)](https://mp.weixin.qq.com/s/8d1qXTm5v4H94HxAibp1dA).

Wayve call this model a VLAM (vision-language-action model). It improves upon the previous work of [Lingo-1](lingo1.md), which is an open-loop driving commentator, and [Lingo-1-X](https://wayve.ai/thinking/lingo-1-referential-segmentation/) which can outputing reference segmentations. Lingo-1-X extends vision-language model to VLX (vision-language-X) domain. Lingo-2 now officially dives into the new domain of decision making and include action as the X output.

The action output from Lingo-2's VLAM is a bit different from that of RT-2. Lingo-2 predicts traejctory waypoints (like ApolloFM) vs actions (as in FSD).

The paper claims that is is a strong first indication of the alignment between explanations and decision-making. --> Lingo-2 is outputing driving behavior and textual predictions in real-time, but I feel the "alignment" claim needs to be examined further. 


#### Key ideas
- Why language?
	- Accelerates training
	- Offers explanability of E2E one model
	- Offers controllability of E2E one model
	
> Language opens up new possibilities for accelerating learning by incorporating a description of driving actions and causal reasoning into the model’s training. In addition, natural language interfaces could, even in the future, allow users to engage in conversations with the driving model, making it easier for people to understand these systems and build trust.

- Architecture
	- the Wayve vision model processes camera images of consecutive timestamps into a sequence of tokens
	- Auto-regressive language model. 
		- Input: video tokens and additional variables (route, current speed, and speed limit) are fed into the language model. 
		- Output: a driving trajectory and commentary text.
- The input of language allows driving instruction through natural language (turning left, right or going straight at an intersection).

#### Technical details
- The E2E system relies on a photorealistic simulator. [Ghost Gym](https://wayve.ai/thinking/ghost-gym-neural-simulator/) creates photorealistic 4D worlds for training, testing, and debugging our end-to-end AI driving models.

#### Notes
- The blog did not say whether the video tokenizer is better with tokenizing the latent space embeddings after a vision model or directly tokenize the raw image (like VQ-GAN or MAGVIT). It would be interesting to see an ablations study on this.
- If language is taken out from the trainig and inference process (by distilling into a VA model), how much performance loss would Lingo-2 lose? It would be interesting to see an ablation on this as well.

