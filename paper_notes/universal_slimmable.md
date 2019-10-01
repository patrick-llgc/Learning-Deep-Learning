# [Universally Slimmable Networks and Improved Training Techniques](https://arxiv.org/abs/1903.05134)

_September 2019_

tl;dr: Extend [slimmable network](slimmable_networks.md) to arbitrary width in a context of channel-wise residual learning.

#### Overall impression
This paper is gold! Lots of insights into the training of neural network.

The practice into calculating the batch norm stats after training is quite enlightening. Refer to the [presentation by Yuxin Wu from FAIR](../assets/papers/Devils_in_BatchNorm_yuxin_wu.pdf).

The universally slimmable network can be trained to achieve similar or slightly better performance with slimmable networks with fixed number of switches. 

Essentially training progressively smaller/narrower networks serve as deep supervision of the entire network.

#### Key ideas
- Each channel can be seen as a residual component (in analogy to network depth). --> wider networks should have better accuracy.
- In slimmable networks, the training time largely scales with the number of switches in the network. This is not scalable for training if we want arbitrary width during deployment. One important role of training is to accumulate dataset stats --> Calculate BN stats of all width after training. A randomly sampled batch (1024 images) already gives very good performance. 
- During training, random sample 2 width + min + max width for training ("sandwich rule" 夹逼定理). (n=4 is the best with diminishing returns when n increases)
- **In-place distillation**: use max width's prediction as soft label to train narrower networks. For largest/widest network, the gt is used. --> this siginificantly improves the performance without additional cost (just replacing hard gt with predicted soft gt)
- The smaller the lower bound, the worse the general performance of networks --> This is the inefficiency in the current training scheme?

#### Technical details
- Review of batchNorm: during training it normalizes features with mean and variance of current mini-batch, while in inference, moving averaged statistics of training are used instead. 
- The authors used a method to calculate the average of streaming data, simply with momentum m = (t-1)/t.
- The prediction is used as soft targer after detach the gradient y' = y'.detach() to avoid gradient from training smaller network contaminate the bigger networks.
- Training smallest network is more important than training the largest network, if we have to pick one. 
- Averaging the output by input channels improves performance slightly and used as default.

#### Notes
- Questions and notes on how to improve/revise the current work  

