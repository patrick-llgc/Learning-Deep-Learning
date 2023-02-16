# [Codex: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)

_February 2023_

tl;dr: Improved version of GPT3 trained on code. Model behind Github Copilot. 

#### Overall impression
This paper introduces Codex, the model behind [Copilot](https://github.com/features/copilot). It can synthesize programs from docstrings (and the reverse). Better performance can be done by generating multiple results and picking the one that is the best.

Codex has the same architecture as GPT-3, but finetuned with code data. It has developed the CoT (chain of thought) and significantly boosted the performance on solve code problems. 

As a technical report from openAI, the paper also has a lengthy discussion on the limitations, risks and mitigation plans, from their commitment to responsible and safe AI. 

#### Key ideas
- The topK idea of "generate more first, then rank and select" is very similar to prediction task of generating more than needed and select the best. 
- GPT lineage
	- GPT3 (0)
	- Codex finetuned on code (28.8%)
	- Codex-S from Supervised finetuning (on correctly implemented standalone functions) (37.7%)
- **Functional Correctness**
	- match based metrics: BLEU score
	- functional correctness is inspired by TDD (test driven development) and uses a set of unit tests to tell if a function implementation is correct. 
	- Functional correctness does not correlate with BLEU scores. The BLEU scores of correct and wrong answers are not cleanly separable. 
	- pass@k score, a problem is considered solved if at least 1 of k code samples passes the unit tests. **Repeated sampling** is an efficient strategy.
	- HumanEval contains a set of 164 hand-written programming problems, with 7.7 unit tests per problem.
- Training and inference
	- training data contains 159 GB of unique python code under 1MB from github.
	- Finetuning and train from scratch has the same performance.
	- During inference, tokens were sampled from Codex until one stop seq is encountered (such as \nclass, \ndef, etc)
	- It is important to optimize **sampling temperature** --> What is this?
	- Choosing the sample with the *highest mean token log prob* outperforms evaluating a random sample.

#### Technical details
- **Data efficiency**: Codex is not sample efficient to train. It is trained on hundreds of millions of line of code. 
- **Misalignment**, or alignment failure, happens when there is some task that X that we want the system to do, and it is capable of doing X, but "chooses" not to. The misalignment gap increases with model size. For example, when the prompt contains subtle bugs, Codex tends to generate poor code than it is capable of.
- Codex can be used to create diverse software that accomplishes the same tasks. This may spell challenges to antivirus systems. 
- To handle prompts of varying length in a batch, shorter prompts are left-padded to the length of the longest patch, so the first token in the reference solution (GT) is aligned. 
- Evaluate pass@k
	- to evaluate pass@k, we generate n>k sample (n=200, k<100), and count the correct samples c. $pass@k = \mathbb{E} [1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}]$
- A conjugate model for docstring generation is trained. It can be used for safety reasons to describe the intent behind generated code. 
	- training data is function signature + reference solution, then the docstring.
	- Evaluation is hard to automate and thus done by human annotators. 
	- The most common failure modes are lack of specific details or over-conditioned on function names.
	- Docstring quality is low, as coders tend to devote less time to writing docstrings. Poor quality examples include "I just found this functional online", etc. --> GIGO.

#### Notes
- What is T (sampling temperature)?
