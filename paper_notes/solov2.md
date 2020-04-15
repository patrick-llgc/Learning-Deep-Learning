# [SOLOv2: Dynamic, Faster and Stronger](https://arxiv.org/abs/2003.10152)

_April 2020_

tl;dr: Dynamic mask kernel for each object in [SOLO](solo.md).

#### Overall impression
This work builds on [SOLO](solo.md) and takes the **decoupled SOLO** idea one step further by predicting the filters dynamically.

The paper proposed two main improvements: Matrix NMS to speed up NMS on masks, and predicting dynamic kernel weights.

Matrix NMS addresses the issues of hard removal and sequential operations at the same time. 

The idea of a splitting the masks head to mask feature branch + mask kernel branch is essentially very similar to the prototype masks (mask features) and  coefficients (mask kernel) in [YOLACT](yolact.md).

![](https://pic2.zhimg.com/80/v2-005de3343859a9c64a3b2d9cc9abc1a9_1440w.jpg)

#### Key ideas
- **Dynamic kernel** (conv filter weights) 
	- In [SOLO](solo.md), in the last layer of the mask branch, the feature map is F of size $H \times W \times E$ (embedding), and we learn the mask kernel G of size $1 \times 1 \times E \times S^2$. This is essentially $S^2$ classifiers, each predicting a segmentation map of size $H \times W$ indicating if a pixel belongs to this cell location category. 
	- Among the $S^2$ cells only a small fraction is used during inference. No need to predict all of them. 
	- For each grid cell, predict D coefficients (D=E if we use 1x1 conv, D=9E if use 3x3 conv). <-- This idea is actually quite similar to [YOLACT](yolact.md).
- **Matrix NMS**: faster and better, combination of [fast NMS](yolact.md) and soft NMS. 
	- Tell how much confidence of itself is decayed by other higher scoring masks. 
	- Tell if higher scoring masks themselves are decayed themselves. If so, their impact on the current box/mask is discounted accordingly.
	- The whole idea is somewhat similar to TF-IDF in NLP. 

#### Technical details
- From visualization of mask feature branch, we can see these are similar to the prototype masks, but cleaner. The mask features are positional sensitive. 

#### Notes
- What stops us from learning all the weights but perform inference only on the selected ones? This way we can speed up the inference without changing SOLO's elegant design. 

