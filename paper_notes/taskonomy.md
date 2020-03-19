# [Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/abs/1804.08328)

_March 2020_

tl;dr: Answers this question: which tasks are better transferred to others?

#### Overall impression
This paper proposed a large dataset with 4 million images, each has 26 tasks labeled as GT. This work directly inspired [task grouping](task_grouping.md), which answers a different question of how to perform multitask learning more efficiently.

The main purpose of transfer learning is to reduce the number of images needed for training the task--they focus on supervision efficiency. Given enough images, trained from scratch is also viable, per [rethinking ImageNet pretraining](rethinking_pretraining.md).

#### Key ideas
- A fully computational approach to reveal the relationships between tasks. Previously the relationship is based on human intuition or analytical knowledge.
	- Depth -> normals are easy for humans, but the opposite is true for NN.
- Four main steps
	- Task specific modeling
	- Transfer Modleing (frozen encoder, trained shallow transfer+decoder)
	- Normalization with AHP (analytical hierarchical process): only assumes monotonocity and keeps the ordinal order
	- Taxonomy extraction with BIP (boolean integer programming) with constraints

#### Technical details
- With only 2% of data, finetuning from reshading to surface normal yields good results already.
- Architecture
	- The backbone is ResNet-50
	- Neck is 2 conv layers (concat channels for higher-order tasks
	- Decoder: 15-layer fully convolutional network, 2-3 FC layers. 
- Transitive transfer is multi-hop transfers.

#### Notes
- [Video of the presentation at CVPR](https://www.youtube.com/watch?v=9mdCWMVAMLg)
- What if I have a new task and I want to quantify which other tasks are related to this task?
	- Need 2k images to find which are the best sources for it
	- Use pretrained networks and use 2k images to finetune
- [Beam search](https://www.youtube.com/watch?v=RLWuzLLSIgw)
	- Beam search is an efficient way to explore a graph. Beam search is not optimal (that is, there is no guarantee that it will find the best solution).
	- Beam search with beam width B=∞ is essentially breadth-first search.
	- Beam search with beam width B=1 is essentially greedy search.
- Powerset of any set S is the set of all subsets of S, including the empty set and S itself, variously denoted as $\wp (S)$.

