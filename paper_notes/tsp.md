# [TSP: Rethinking Transformer-based Set Prediction for Object Detection](https://arxiv.org/abs/2011.10881)

_November 2020_

tl;dr: Train DETR faster and better with sparse attention and guided Hungarian matching.

#### Overall impression
The paper digs into the reasons why [DETR](detr.md) is so hard to train, primarily the issues with Hungarian loss and the Transformer cross attention mechanism. The paper proposed two improved version of DETR based on existing solutions, TSP-FCOS (improved [FCOS](fcos.md)) and TSP-RCNN (improved Faster RCNN).

This work basically borrows the best practice in modern object detector (FPN, FCOS and Faster-RCNN) and replaces the dense prior heads with DETR encoder, and uses set prediction loss. 

There are several papers on improving the training speed of [DETR](detr.md).

- [Deformable DETR](deformable_detr.md): sparse attention
- [TSP](tsp.md): sparse attention
- [Sparse RCNN](sparse_rcnn.md): sparse proposal and iterative refinement

A standard 36 epochs (3x schedule) can yield a SOTA object detector. 

> Object detection is essentially a set prediction problem, as the ordering of the predicted objects is not required. Most modern object detectors uses a **detect-and-merge** strategy, and makes predictions on a set of dense priors. The dense priors makes NMS necessary. The detection model is trained agnostically wrt the merging step, so the optimization is not end-to-end and arguably sub-optimal.
> 
> DETR removes the handcrafted parts such as dense prior design, many-to-one label assignment problem and NMS postprocessing.
>
> DETR removes the necessity of NMS as self-attention component can learn to remove duplicated detection. The Hungarian loss encourages one target per object in the bipartite matching. 


#### Key ideas
- Cross attention in decoder impedes convergence
	- Transformer attention maps are nearly uniform in the initialization stage but becomes increasingly more sparse during the training process toward convergence. 
	- Sparsity of cross attention consistently increases even beyond 100 training epochs. It is super hard to train.
	- Encoder self-attention is essentially identical to the self-attention in a non-autoregressive decoder, so a set prediction scheme is still feasible for encoder-only DETR. --> Just that no query is needed anymore, and all predictions are wrt the input feature point or region of interest (FOI or ROI).
- New bipartite matching scheme
	- Instability in bipartite matching in original DETR affects convergence, but a matching distillation experiment reveals that this is not the bottleneck after 15 epochs
	- **Faster set prediction loss**: borrows GT assignment rules from FCOS and Faster RCNN. For TSP-FCOS, A feature point can be assigned to a GT only when the point is in the bounding box of the object and in the proper feature pyramid level. 
- TSP-FCOS
	- FoI classifier: the computation complexity of transformer's self-attention module scales quadratically wrt the sequence length, quartically with image size. A binary feature point classifier selects a limited portion of features (FoI). 
	- FoI and their positional embedding feed into the encoder. Self-attention module is used to aggregate information of different FoIs. 

#### Technical details
- Sparsity of attention map is measured in terms of negative entropy
- The input of encoder head: 700 top-scoring FoI from FoI classifier are selected in TSP-FCOS, and 700 top-scoring RoI from RPN are selected in TSP-RCNN.
- During training, 70% of inputs to encoder is randomly dropped to improve robustness of set prediction.

#### Notes
- Questions and notes on how to improve/revise the current work  

