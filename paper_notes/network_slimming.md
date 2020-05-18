# [Network Slimming: Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)

_May 2020_

tl;dr: Channel pruning by learning with L1 sparse constraint on batch norm.

#### Overall impression
This paper proposes a simple idea of gamma (channel scaling factor) decay. It adds a L1 sparse constraint on BN scale parameter gamma. During inference time, when gamma is smaller than a global threshold, set the entire channel to zero.

#### Key ideas
- Advantages of network slimming:
	- No need to change architecture. Recycles the parameters in BN. 
	- No need for special lib for inference.
- The pruning can happen multiple pass.
- Sparsity constraints can actually help performance, even before pruning.
- During training, the channel scaling factor can actually go up or down.
![](https://media.arxiv-vanity.com/render-output/3010938/x6.png)

#### Technical details
- Un-structured pruning can only save model size by storing it with a sparse format.
- In Batch Norm, there are 4*C parameters (2 trainable and 2 un-trainable parameters per channel).
![](https://miro.medium.com/max/1067/1*ETvcPhYH1lCfXndMiKW-jQ.png)
- The additional L1 regularization term rarely hurt model performance
- ResNets are harder to prune. Only 10% or so can be pruned away without hurting performance after a model is trained and without further finetuning. (per [pruning filters](pruning_filters.md)). The paper also reported that for ResNet-164, the FLOPs saving ratios is around 50%. 

#### Notes
- [code on github](https://github.com/Eric-mingjie/network-slimming)
- How is the **channel selection layer** implemented for architectures with cross layer connection such as ResNets?