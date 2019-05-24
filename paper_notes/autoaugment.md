# [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)

_May 2019_

@YuShen1116

tl;dr: Use reinforcement learning to train a agent to search data augmentation policy. 

#### Overall impression
Instead of hard code the data augmentation(image flip, resize, rotation), they trained a small network(reinforcement learning) to decide which data augmentation should be applied on the training data. They also tried auto augmentation transferring(trained on CIFAR-10 and applied on difference dataset for instance.) and the results show that transferring could also work.

#### Key ideas
- They define data augmentation as a search problem in the operation space(detail in later).
- The search policy contains 5 sub-policies with each sub-policy consisting of two image operations. each operation is associated with two hyper-parameters: 1. probability of applying the operation. 2. the magnitude of the operation.


#### Technical details
- They have total 16 image operations(such as shear X/Y, Rotate, AutoContrast...) in the search space and each operation's magnitude is discretized into 10 values.
- The probability of applying a operation is discretized into 11 values.
- Finding each sub-policy becomes a search problem in a space of (16 x 10 x 11)^2 possibilities. (16 x 10 x 11)^10 for 5 sub-policies
- The search algorithm contains two parts: a controller(RNN) and training algorithm(Proximal Policy Optimization algorithm)
- At each step, the controller's prediction is fed into the next step as an embedding(operation type, magnitude, probability).
- RNN is a one-layer LSTM with 100 hidden units at each layer and 2 X 5B softmax prediction for the two convolutional cells.(B is typically 5).
- Training procedure (not very sure about this part)
    - For each training mini-batch, one of the 5 sub-policies is chosen randomly to augment the image.
    - evaluate this child-model on the validation set and use is as the reward signal to train RNN.
- At the end of the search, concatenate the sub-policies from the best 5 policies into a single policy(with 25 sub-policies), and use it to train the model for dataset.

#### Notes
- Not really understand how to concatenate the sub-policies from the best 5 policies, where does those 5 policies come from? Aren't them sub-policies? 
- On CIFAR-10, AutoAugment(pre-trained on reduced CIFAR-10) picks mostly color-based transformations, on SVHN, AutoAugment(pre-trained on reduced SVHN) would choose invert, Shear, Rotate.
- It's very interesting that the agent learn to choose different augmentation policies for different dataset. For instance, SVHN dataset contains image such as 
door plate number, it's won't change the model's generalization ability with change the color, but augmentation such as 
invert, rotate would work.

