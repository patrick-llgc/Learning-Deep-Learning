# [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

_February 2023_

tl;dr: Align LLM with RLHF.

#### Overall impression
This paper proposes InstructGPT, the backend model that powers [ChatGPT](https://openai.com/blog/chatgpt/). It provided a practical way of making products with generative model.

The paper showed that from proper finetuning with extra data and RL magic with human feedback, instructGPT can generate results strongly preferred by human evaluation with 100x less parameters. Finetuning with human feedback is a promising direction for aligning language models with human intent. 

Mis-alignment is a serious issue for LLM, so much so that openAI even has a dedicated team to tackle it. This work is from openAI Alignment Team. We wan the LLM to be "helpful, honest and harmless". InstructGPT is trained to maximize helpfulness, but evaluated for honesty/truthfulness and harmlessness. 

The paper is more like an experiment report, without much scientific novelty. Yet it meticulously demonstrates the details of this ground-breaking engineering feat. 

#### Key ideas
- RLHF uses human preferences as a reward signal to finetune the model.
- ChatGPT/InstructGPT did not invent the methodology RLHF. The same methods have been used before to align text styles (also by openAI). 
- The 3 step process in RLHF (reinforcement learning with human feedback)
![](https://cdn.openai.com/chatgpt/draft-20221129c/ChatGPT_Diagram.svg)
	- SFT model finetuning GPT-3 using labeled data. 第一步，手动选择一些问题并以人工的方式给出答案，以上述作为数据集来训练SFT模型
	- RM model training to give a scalar reward for each output from SFT model. 第二步：让训练好的SFT模型回答一些问题，人工对答案进行打分，然后以这一部分数据集来训练RM模型
	- Policy model training with PPO. 第三步，根据RM模型的打分结果，利用强化学习继续优化SFT模型
- 3 datasets
	- SFT 13k training prompts, with labeler demonstration
	- RM: 31k training prompts, with labeler ranking 
	- PPO: 33k training prompts, collected from API, without human label
- InstructGPT achievements vs GPT3
	- Highly preferred
	- Improved truthfulness (less hallucination)
	- Small improvement of Toxicity, not bias.
	- Less **alignment tax** through modified RLHF training
	- Generalizes to held-out human labelers
	- NLP dataset does not reflect how LM are used. InstructGPT perform slightly worse than SFT baseline on many task baselines
	- More **prompt-friendly** and requires less careful prompting.
	- RLHF is highly effective. GPT-3 uses 3600 petaflops/s-days (8x A100 delivers [5 petaflops](https://nvidianews.nvidia.com/news/nvidia-ships-worlds-most-advanced-ai-system-nvidia-dgx-a100-to-fight-covid-19-third-generation-dgx-packs-record-5-petaflops-of-ai-performance), [link](https://twitter.com/id_aa_carmack/status/1192513743974019072?lang=en)). SFT uses 5, and PPO-ptx uses 60. This would take 720 8-card-node*day to finish.
- RM model
	- Online feedback is hard as it requires human labeling after every model update. We need to build an "environment" for the policy model to interact with. Need to train a model to replace human.
	- Train a RM (reward model) to rate the output of a policy model. 
	- RM takes in prompt and response, and output a scalar reward.
	- Train on all (K select 2) comparisons from each prompt as a single batch element. This saves forward passes and speeds up training. In a batch, average all possible (K select 2) pairs, each pair with a CE loss term.
	- RM model uses **Pairwise Ranking Loss**. This is improved from previous method of picking the best 1/N which is prone to overfitting.
- RL training
	- PPO loss with RM reward, and KL div loss term from SFT model to mitigate over optimization of the reward model.
	- PPO-ptx also mixes the pretraining weights to reduce the "alignment tax". InstructGPT is the PPO-ptx model.
- Eval
	- GPT3 can be improved by using well-crafted few-shot prompt
	- Then improved by training on demonstration with SFT
	- Then training on comparison data with PPO
- Cons
	- When given an instruction with a false premise, the model sometimes incorrectly assumes the premise is true. 
	- InstructGPT can overly hedge rather than directly answering simple questions. (车轱辘话)
	- Performance degrades when the instructions contains multiple explicit constraints. 
	- InstructGPT can generate more toxic output than GPT-3, when prompted so.

	
![](https://miro.medium.com/v2/resize:fit:1200/format:webp/1*--3MnQ-ktKKTddi-QzFrvA.png)

#### Technical details
- Labeling and labeler screening. It is easy to hire someone, but hard to hire someone to generate high quality annotation. The paper includes very detailed labeler screening tests and even labeling tool UI. Similar to [MEBOW](mebow.md).
- Train, evaluation and test split based on user ID to avoid info leakage.
- Dataset 96% English, but generalizes well to other languages and coding tasks. 
- Labelers are carefully screened and instructed, via Upwork and ScaleAI. For evaluation labelers they do not undergo screening test. Inter-annotator agreement rate is around 75%.
- Ranking/comparison may not be the most efficient way to providing alignment signal. There is a vast space of options for designing interfaces for labelers to provide feedbacks to LLM.

#### Notes
- In easy tasks, [GPT-3](gpt3.md) demonstrated that LLM is zero-shot learner with good generalization, but [Codex](codex.md) and [InstructGPT](instructgpt.md) showed that in more complex tasks, finetuning with dedicated dataset may be necessary. 
- InstructGPT only did the human in the loop iteration once. Actually this could happen in an iterative way.
- Difference between toy and tool. Tools have to be reliable, without surprises. Toys are to surprise, and is allowed to make stupid mistakes. This comparison pattern is also reflected in production vs. demo.
- Tons of LLM have failed to productize due to PR disasters (Google's Gorilla, Facebook's Primate, MSFT's Tay, Meta's Galactica). ChatGPT can make such a big splash because 1) it is created by a startup and people are more lenient to its errors. 2) It has taken enough measures to curb its rough edges (toxic, biased replies), so much so that people can discovering its beauty before shutting it down due to PR crisis. 
- The methodology used in generative models can be possibly applied to prediction and planning task in autonomous driving. The RM model can be also used for rating. Very much like the relation predictor in [M2I](m2i.md).
- [Review by Li Mu 李沐 on Bilibili](https://www.bilibili.com/video/BV1hd4y187CR)
	- Tradeoff of engineering complexity: RL leads to more complex system and instability. Maybe we can label a larger amount of data and use it to finetune.
	- 从技术上来讲，InstructGPT还是一个挺实用的技术，它告诉了大家一个方法，说给定一个比较大的语言模型，你怎样通过一些标注数据，能迅速地把它在某一个你关心领域上的性能提升，使得它能达到一个实用的阶段。如果大家想用这种生成模型做产品，这篇文章就提供了一个实际可操作的思路。