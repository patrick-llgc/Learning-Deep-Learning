# [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

_February 2020_

tl;dr: Use momentum updates to improve upon previous work on unsupervised pretraining.

#### Overall impression
The whole field of unsupervised visual representation learning is gaining more attention due to recent success in unsupervised pretraining in NLP (BERT, GPT). The framework of the problem is not new, and the paper largely inherits the set up from previous work (e.g. [InstDisc](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0801.pdf)). This paper draws a lot of inspiration from InstDisc paper, and can see as an immediate update to it.

The main contribution of this paper is the update of the encoder of the dictionary. InstDisc proposed the dictionary which decoupled the batch size and the dictionary size. However it maintains and updates the representation of each instance separately. MoCo instead updates the encoder. The momentum updating rule reminds me of the target network trick in [DQN](nature_dqn_paper.md). (The trick of caching a historical version of the model is also mentioned in the discussion thread in [知乎](https://www.zhihu.com/question/355779873/answer/894687533))

#### Key ideas
- In the field of unsupervised pretraining, there are two tasks: **pretext** task (used for pretraining, such as image cls) and **downstream** task (that are finetuned from the pretrained weights, such as object detection).
- Build a large and consistent dictionary. The dictionary is kept as a queue of samples. The key is embedding mapped by an encoder. The key encoder needs to be slowly updating.
- Contrastive loss: when q is similar to positive key and dissimilar to all other keys (Noise-contrastive estimation, NCE) --> this is yet another example of assigning loss without knowing GT and prediction assignment, and depends on the differentiability of min/max function.
- End-to-end training performs really well and sets the benchmark for MoCo. It updates encoder for query and dict key. However the consistent dict size is limited to minibatch size. 

#### Technical details
- Pretraining benefits from a large K. So keep a queue to increase the dictionary size is critical.
- Pretext task: two random views of the same image through augmentation as positive pair and all the rest as negative (followinbg InstDisc).
- Shuffling BN: normal BN leaks info between different batches and cannot be used here. 
- 128-D vector per image.

#### Notes
- Questions and notes on how to improve/revise the current work  

