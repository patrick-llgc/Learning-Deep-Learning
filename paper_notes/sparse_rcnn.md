# [Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450)

_November 2020_

tl;dr: Summary of the main idea.

#### Overall impression
This paper rethinks the necessity of dense priors (either anchor boxes or reference points) in object detection, very similar to [TSP](tsp.md). Sparse RCNN uses a number of sparse proposals (N << HWk dense priors) for object detection.

There are several papers on improving the training speed of [DETR](detr.md).

- [Deformable DETR](deformable_detr.md): sparse attention
- [TSP](tsp.md): sparse attention
- [Sparse RCNN](sparse_rcnn.md): sparse proposal and iterative refinement

The iterative head design is quite inefficient in capturing the context and relationship with other parts of the image, and thus needs quite a few iterations (~6 cascaded stage). In comparison, the sparse cross attention in [Deformable DETR](deformable_detr.md) and [TSP](tsp.md) may be a better way to go. 

#### Key ideas
- Sparse-in and sparse out
	- DETR uses sparse set of object queries to interact with global (dense) image feature. It is also dense-to-sparse.
	- Sparse RCNN proposes both **sparse proposals** and **sparse features**
		- Sparse RCNN uses only 100 proposals, same as [DETR](detr.md). 300 proposal adds very little inference cost due to light design of dynamic head. 
	- Sparse Proposal feature: High dim (256-d) latent features encoding pose and shape of instances. Proposal feature generates a series of customized parameters for its exclusive object recognition head via dynamic conv. 
- Three Input:
	- Image
	- A set of learned proposal boxes (fixed, and dataset independent, and may not be symmetric or shift-equivariant) --> This looks to be an inefficiency that can be optimized. Maybe something like the FoI in [TSP-FCOS](tsp.md).
	- A set of learnable proposal features. The initial values of the proposal feature is actually not that important, given the auto-regressive iterative refinement scheme (similar to IterDet).
- Dynamic instance interactive head --> More like a trick and an afterthought rather than intuition inspired design. 
	- The interaction between Proposal feature and RoI features is modeled by a dynamic convolution. The parameters are generate by the proposal features. 
	- Newly generated object boxes and object features of the next stage in iterative process. Features need to be RoI aligned again, similar to [Cascade RCNN](https://arxiv.org/abs/1712.00726) <kbd>CVPR 2018</kbd>. This only introduces a marginal computational overhead as the dynamic conv is very light.

#### Technical details
- Set prediction loss exactly the same as DETR.
- Ablation studies
- The paper has a very clear evolution path of model, in Table 2, 3 and 4. According to the author's reply on [Zhihu知乎](https://zhuanlan.zhihu.com/p/310058362).

> 
- 18.5到20.5：加上cascade r-cnn那样的结构
- 20.5到32.2: cascade结构 + 上一个stage的obj feature concat到下一个stage的obj feature
- 32.2到37.2: cascade结构 + 上一个stage的obj feature先self-att，再concat到下一个stage的obj feature
- 37.2到42.3: cascade结构 + 上一个stage的obj feature先self-att，在作为proposal feature与下一个stage的RoI做instace interaction

- Initialization of proposal box does not matter much
- Number of proposals from 100 to 500 improves performance, but also requires longer training time
- Stage increase saturates around stage 6. The iterative heads gradually refines box and removes duplicate ones. 

#### Notes
- [Code on github](https://github.com/PeizeSun/SparseR-CNN)
- The Sparse RCNN paper is not very well written. Luckily the first author clarified most of the important details in [Zhihu知乎](https://zhuanlan.zhihu.com/p/310058362). 
- The proposal can be seen as an averaged GT distribution. This should be improved to be data dependent, and the authors are working on a v2 of Sparse RCNN. The authors also argue that a reasonable statistic can already be qualified candidates. Maybe this is similar to the FoI classifier in [TSP-FCOS](tsp.md).

