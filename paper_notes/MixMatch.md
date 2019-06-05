# [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)

_June 2019_

@YuShen1116

tl;dr: A method for unsupervised learning on image classification.

#### Overall impression
The results are very good, the model can achieve 11% error rate on CIFAR-10 with only 250 labels.
Also the new hyper-parameter for mix-match can be fixed for different dataset. 

#### Key ideas
- Given a batch of X labeled data and a batch U of same size unlabeled data. Apply similar computation procedures and different loss functions.  
- Whole process can be divided into 4 parts: data augmentation, label guessing, sharpening and mix-up
    - Data augmentation: applied on both labeled and unlabeled data(didn’t find which augmentation was applied, assuming its like a common tools such as resize, whitening such and such
	- Label Guessing: For each unlabeled example in U, mix-match produces a guessed label for loss term. The guessed label are the average of the model's predicted class distribution across all the K augmentations.
	- Sharpening: a sharpening function is applied to reduced the entropy of label distribution.
	- Mix-up: mix-up is a method from other paper, need to track it. From the ablation study, this step is the most important one. In the original mix-up paper, use separate loss terms for labeled and unlabeled data will cause issue(what issue?). But now in mix match they defined a slightly modified mix-up to solve the problem. 
-Loss function: For labeled data, they used popular cross-entropy loss. For unlabeled data, they used squared L2 loss.
-Hyper parameters: Even though they are many parameters such as strength in sharpening, number of data augmentations. But they states that those parameter can be fixed for different dataset.

#### Notes
- I've seen people saying that this paper is very similar to https://arxiv.org/abs/1904.12848
- There is no experiments on large dataset such as ImageNet.
- This paper is more like combining current popular methods for better results instead of strong slights.