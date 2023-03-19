# [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

_March 2023_

tl;dr: Multimodal GPT with significantly improved, near human-level performance on various academic and professional benchmarks.

#### Overall impression
GPT-4 is a large multimodal model capable of processing image and
text inputs and producing text outputs.

Over GPT-3.5, it is much more capable (through pretraining on a larger scale), more reliable and truthful to fact (through RLHF), and generates less toxic results (through model assisted safety pipeline by adding more safety prompts and rule-based RM).

The technical report did not report the model size due to competitive landscape and safety implications. We know that it uses publicly available data and data licenced from 3rd parties. Also it uses post-training RLHF as in previous work, such a [InstructGPT](instructGPT.md).

It is exciting to see that "GPT-4 was used for help with wording, formatting, and styling throughout this [technical report]."

#### Key ideas
- Capabilities: Human-level performance
	- simulated bar exam, GPT-4 achieves a score that falls in the top 10% of test takers. This contrasts with GPT-3.5, which scores in the bottom 10%
	- The model’s capabilities on exams appear to stem primarily from the **pre-training process** and are not significantly affected by RLHF post-training alignment.
- Limitation: **Truthfulness** (reliability)
	- GPT4 still is not fully reliable (it “hallucinates” facts and makes reasoning errors)	
	- Regarding fact check (on dataset like TruthfulQA) The GPT-4 base model is only slightly better at this task than GPT-3.5; however, after **RLHF post-training** we observe large improvements over GPT-3.5 (roughly from 60% to 80% in internal eval).
	- The model can be **confidently wrong.** The pre-trained model is **highly calibrated** where its confidence in an answer generally matches the probability of being correct). The post-training hurts calibration significantly.
- Risk ans safety concerns: **Toxicity**/Refuse unsafe inputs
	- Two ways to reduce toxicity is through Model assisted safety pipeline, which is to make the model itself less toxic, by 1) adding more data, and 2) improve training method.
	- 1) More data: An additional set of safety-relevant RLHF training prompts were added to training data. Adversarial testing and red-teaming: over 50 experts from domains to adversarially test the model. 
	- 2) RBRMs (Rule-based reward models): rewarding model by refusing unsafe prompts, and reward by not refusing safe prompts.
	- System level protection: model-level interventions increase the difficulty of eliciting bad behavior but doing so is still possible. Therefore, it’s important to complement them with deployment-time safety techniques like monitoring for abuse as well as a pipeline for fast iterative model improvement. --> we need to improve both the model and the system to improve the usability of a AI-centric system, like autonomous driving.


#### Technical details
- Model performance prediction across training scales
	- A key challenge and a core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4’s performance based on models trained in similar ways but with no more than 1/1,000th the compute of GPT-4. 
	- The fitted scaling law predicted GPT-4’s final loss with high accuracy
	- $L(C) = aC^b + c$. L can be both loss, and expected pass rate. 
	- We chose to look at loss because it tends to be less noisy than other measures across different amounts of training compute.
- The model-assisted safety pipeline can have FP by refusing innocuous requests or excessively hedging
- Multilanguage capability
	- GPT-4 outperforms the English-language performance of GPT 3.5 and existing language models for the majority of languages we tested, including low-resource languages such as Latvian, Welsh, and Swahili
- RBRM: 
	- The RBRM takes three inputs: the prompt (optional), the output from the policy model, and a human-written rubric (e.g., a set of rules in multiple-choice style) for how this output should be evaluated.
	- we can provide a rubric that instructs the model to classify a response as one of: (a) a refusal in the desired style, (b) a refusal in the undesired style (e.g., evasive or rambling), (c) containing disallowed content, or (d) a safe non-refusal response
- Improved **safety metrics**: GPT-4 produces toxic generations only 0.73% of the time, while GPT-3.5 generates toxic content 6.48% of time.



#### Notes
- How is predicting large-scale training performance useful? Maybe it is necessary for budgeting, in order to reach a certain KPI goal? Although loss scales more smoothly, it is not easy to connect loss to KPI. 
- The paper listed interesting org chart of OpenAI. Please refer to [this note](../openai_orgchart/README.md) for further details.