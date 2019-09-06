# [Slimmable Neural Networks]()

_September 2019_

tl;dr: Train a single network and deploy differently depending on the HW. 

#### Overall impression
The idea is actually quite similar to that of universal model. In both cases, the different models share the convolutional layers and have switchable/selectable BatchNorm layers. 

In the Slimmable Networks, S-BatchNorm records the different scales and variances of features for each switch. For universal models, the BatchNorm contains the stats for different datasets (which can be vastly different, such as medical imaging datasets and imagenet). 

The training procedure prioritizes the first 25% of channels as compared with later channels. This is quite similar to the idea of the pruning method [LeGR](legr.md). Maybe they can be combined?

The authors also showed that 4-switch and 8-switch variants are actually on the same Pareto Front. 

#### Key ideas
- Why do we need slimmable networks? We may want to train the same network and deploy to different platform. Resources vary even on the same platform. 
- Maintain a list/dict of BatchNorm layers and switch them according to the desired width (Number of channels) of the network. 
- Iterate over all switches in each training iteration. 
- The first 25% of the channels are trained almost 4 times as compared to the last 25% of the channels. Thus the first 25% of channels are more important by design via the training scheme.
- Most slim models improved performance, through implicit model distillation and richer supervision.

#### Technical details
- Reducing depth does not reduce memory footprint. Reducing width does. 
- Incremental training (freezing portion and train the rest) does not work very well. So slimmabel networks are doing more than just learn the residual.
- Quick review: 4 parameters in a BN layer for each channel, 2 learnable and 2 moving averaged stats. 
- BatchNorm layers can somewhat repurpose the convlayers. The same channel in slimmable network may be slightly repurposed based on different switches/width.
- During training, the naive approach (wi the same set of BatchNorm) almost performs the same as S-BN. This is because **during training the stats of the current minibatch is used. The moving averaged stats are only used in tests.**

#### Notes
- Combine LeGR and slimmable nets?