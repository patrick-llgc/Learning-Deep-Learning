# [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf)

_Mar 2019_

tl;dr: Linear blending of labels boosts accuracy and generalizability of classification.

#### Overall impression
The mixup technical is simple as it does not require domain knowledge (data agnostic), but it is surprisingly effective. It can be seen as a special form of data augmentation, and also as a regularization technique.

#### Key ideas
- In essence, mixup trains a neural network on convex combinations of pairs of examples and their labels. By doing so, mixup regularizes the neural network to favor simple linear behavior in-between training examples.
- ERM vs VRM: 
	- ERM: Neural networks are trained as to minimize their average error over the training data, a learning rule also known as the Empirical Risk Minimization (ERM) principle. However, ERM is unable to explain or provide generalization on testing distributions that differ only slightly from the training data.
	- VRM: The method of choice to train on similar but different examples to the training data is known as data augmentation, formalized by the Vicinal Risk Minimization (VRM) principle
- Mixup
$$
x' = \lambda x_i + (1-\lambda) x_j \\
y' = \lambda y_i + (1-\lambda) y_j
$$
where x_i and x_j are raw input vectors, and y_i and y_j are one-hot encoded labels.
- Learning with corrupt data: dropout with high dropout prob is the state-of-the-art method, and mixup is even better.
- Mixup increases the robustness to adversarial examples
- Mixup can be used to boost performance of ML algorithms on tabular data
- Mixup stabilizes the training of generative adversarial networks.

#### Technical details
- Ablation studies show that blending input from different classes, and doing label blending (instead of picking the one with higher weights) yielded better results.
  - SMOTE is basically mixup with kNN of the same class. This does not lead to performance increase on CIFAR dataset.
  - Interpolating/Extrapolating the data from tesame class can lead to good performance according to [Dataset Augmentation in Feature Space](https://arxiv.org/pdf/1702.05538.pdf). However this method has hyperparameters that are hard to choose, and if it always helps. In comparison, the mixup method addressed both issues.
- Like label smoothing, the supervision of every example is not overly dominated by the ground-truth label.

#### Notes
- This mixup technique can be expanded to [object detection](bag_of_freebies_object_detection.md).

