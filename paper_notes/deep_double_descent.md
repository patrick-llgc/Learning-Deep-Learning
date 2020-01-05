# [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292)

_January 2020_

tl;dr: Double descent is a robust phenomenon that occurs for various tasks, architectures and optimizers. 

#### Overall impression
This paper extends the work of [double descent](double_descent.md) to deep neural networks. 

The main contribution is that the double descent not only happens for more complex models (increasing num of channels) but also for training epochs. The authors proposed **effective model complexity (EMC)** which is training routine specific to describe this behavior. Increasing training time will increase EMC. 

Also more data may not help in the critical region, leading to sample non-monotonicity.

#### Key ideas
- Deep double descent happens model-wise and epoch-wise. --> **training longer can correct overfitting.**
- EMC: the max sample number for training procedure T (including model arch, optimizer, epochs, etc) to achieve ~0 training error. 
	- EMC extends the interpolation threshold notion in model size to include training epoch as well. EMC can manifest as threshold in model size or training epoch.
- **Hypothesis of deep double descnet**: If num of samples is sufficiently larger or smaller than EMC, then increasing model will leads to better results. Otherwise larger model may hurt. 
- **An intuitive explanation** about deep double descent is that for model-size interpolation threshold, there is only one model that fits the training data and this model is very sensitive to noise and forcing it to learn slightly more noisy/complex data will destroy the global structure. 
- Advices for practitioners:
	- if a training procedure is barely able to fit the training set, then small changes to the model may lead to unexpected behavior. 
	- Early stopping helps alleviates double descent, but not entirely.
	- Double descent is stronger in settings with more label noise, or more "model mis-specification", or a harder data distribution to learn.
	- Ensemble helps greatly in this critical region.
	- How do I know I am in critical region? How do I find if my model is large enough? --> **Open question**
	- Data augmentation shifts interpolation peak to the right. It does not necessarily help when noise level is high. (Fig. 5)
	- conventional wisdom divide training epochs into two regions, underfit and overfit. There may exist a third region when test error dips again.
	- Based on my own observation, deep double descent may happen after really long time. However this is free model capacity that does not impact test time and may be worth a try.

#### Technical details
- Sample non-monotonicity: 
	- increasing samples reduces AUC under test error curve wrt model size
	- more samples moves the curve to the right, increasing the interpolation threshold
	- but for a particular model size the test error may not improve.

#### Notes
- People have discovered in the interpolation threshold, critical phase transition of the loss landscape may happen, such as in this PRE paper [The jamming transition as a paradigm to understand the loss landscape of deep neural networks](https://arxiv.org/abs/1809.09349).
- Blog [Understanding “Deep Double Descent”](https://www.lesswrong.com/posts/FRv7ryoqtvSuqBxuT/understanding-deep-double-descent)
- Blog [Are Deep Neural Networks Dramatically Overfitted?](https://lilianweng.github.io/lil-log/2019/03/14/are-deep-neural-networks-dramatically-overfitted.html) by Lilian Weng

