# [SOLO: Segmenting Objects by Locations](https://arxiv.org/abs/1912.04488)

_April 2020_

tl;dr: Single-shot instance segmentation.

#### Overall impression
The paper proposes a simple frame work for instance segmentation directly. Essentially it is a YOLO architecture predicting additional HxW values at each cell. The HxW values are warped into a mask with the same resolution of the feature map. <-- **However there is an important trick that reshapes the SxSx(HxW) into HxWx(SxS).**

Semantic segmentation classifies each pixel into a **fixed** number of categories. Instance segmentation has to deal with a **varying** number of instances. That is the biggest challenge. Instance segmentation can be sorted into top down approaches such as Mask RCNN and bottom up approaches such as [Associate Embedding](associative_embedding.md).

The **decoupled SOLO** idea is fabulous and I think is partially inspired by [YOLACT](yolact.md) by predicting prototype 2S masks.

This paper can be seen as an extension to the anchor-free object detection, such as [FCOS](fcos.md) and [CenterNet](centernet_ut.md), but with the important trick of reshaping the tensor. <-- See discussion in [TensorMask](tensormask.md).

Direct spatial2channel leads to spatial alignment too poor to guarantee good mask quality. (see natural representation in [TensorMask](tensormask.md)). However it should be enough to guarantee the SxS order. 

#### Key ideas
- **Grid cell**: assumption is each cell of the SxS grid must belong to one individual instance. The instance mask branch has $H \times W \times S^2$ dimension.
- **FPN for multi-level prediction**. Each feature map is only responsible for predicting masks within a certain scale range. 
- **Dice loss** to balance small and large masks. It leads to better performance than cross entropy or focal loss.
- **CoordConv** to introduce spatial variance. (But why?)
- Architecture
	- Center category: similar to [CenterNet](centernet_ut.md) but on a coarser grid.
	- Mask category: for each positive grid, it corresponds to a channel in the HxW size feature map. 
- **Decoupled SOLO**: Predicting $S \times S$ mask is heavy. Instead the decoupled SOLO predicts $2 S$masks and use mask = element-wise multiplication of two masks. This yields the same results as vanilla SOLO.

#### Technical details
- Feature alignment: The backbone has spatial size $H \times W$. There are two heads, one with $S \times S$ and the other with $H \times W$. Resampling/align is needed to map $H \times W \rightarrow S \times S$. Interpolation, adaptive pooling, RoIAlign all generate similar results. 

#### Notes
- What happens if we predict $S \times S$ masks for each grid? --> too coarse.
- What happens if we predict $H \times W$ positions for each position? --> too much computation
- So basically we need a coarse position prediction and a high resolution mask prediction. Thus SxS grid and HW resolution mask. 
- The formulation is essentially prediction a $S \times S \times (C + H \times W)$ feature map from a $H \times W \times Channels$ feature map, with FCN. The loss is done on warped masks and compare with GT.

