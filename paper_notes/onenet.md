# [OneNet: End-to-End One-Stage Object Detection by Classification Cost](https://arxiv.org/abs/2012.05780)

_December 2020_

tl;dr: Easy-to-deploy end to end object detector. Classification cost is the key to remove NMS.

#### Overall impression
This paper is from the authors (孙培泽 et al) of [Sparse RCNN](sparse_rcnn.md). This paper has a focus on deployment: single stage, association with min cost function (instead of Hungarian). This is perhaps one of the best paper written during the boom of end-to-end object detectors. 

Existing approaches (anchor-based or anchor-free) assign labels by location cost only, either IoU based Box Assignment or point distance based Point Assignment. Without classification cost, only localization cost leads to redundant boxes of high confidence scores in inference, making NMS a necessary post-processing component. 

The idea of introducing classification to handle **sudden changes** in the value domain is a quite common practice, as in practice, sudden changes reflects multimodality. For example, to handle the oversmoothing issue in depth prediction, [Depth Coefficient](depth_coeff.md) and [SMWA](smwa.md) introduces classification of multiple models. The regression model responds smoothly to smooth changes in input. Therefore if we only have localization cost as matching cost, we will have duplicate bboxes around GT boxes. 

#### Key ideas
- label assignment: minimum cost assignment, instead of heuristic rule (like in [RetinaNet](retinanet.md) or [FCOS](fcos.md)) or complex Hungarian matching (like in [DETR](detr.md)).
- **Misalignment between label-assignment and network optimization objective** is the main reason why NMS is necessary. 
	- Training objective to min cls + reg. If during assignment only the one minimizing the location cost is the assigned as positive sample. This positive sample minimizes the location cost but not necessarily  This leads to suboptimal situation that there may exist another location which has lower classification
- training loss is similar to matching cost, comprising of focal loss, L1 loss and GIoU loss.
- Head 
	- classification head. Note that this is not necessarily the bbox center. The network learns the most discriminative location for classification. This is possible by removing the heuristic matching strategy.
	- regression head: distance to four edges
- The final output is top-K bboxes (K=100). This just finds the top 100 scoring pixels in the heatmap.

#### Technical details
> [DETR](detr.md) can be viewed as the first end-to-end object detection method; DETR utilizes a sparse set of object queries to interact with the global image feature. Benefiting from the global attention mechanism and the bipartite matching between predictions and ground truth objects, DETR can discard the NMS procedure while achieving remarkable performance. [Deformable-DETR](deformable_detr.md) is introduced to restrict each object query to a small set of crucial sampling points around the reference points, instead of all points in the feature map. [Sparse R-CNN](sparse_rcnn.md) starts from a fixed sparse set of learned object proposals and iteratively performs classification and localization to the object recognition head.

- The paper also noted that [CenterNet](centernet.md) labels only one positive samples per GT, but also labels nearby samples by Gaussian blur. In other words, if we assign one-hot labels to centerNet, it also should be able to learn.
- predicted location cost vs predefined location cost
	- **predicted** location cost means loss calculation based on pred/gt pair matched by min loss matching
	- **predefined** location cost means loss calculation based on pred/gt pair matched by heuristics matching
- From Fig. 6, it seems that without classification, no bbox emerges as scores > 0.4.
- pseudo-code of minimum cost assignment.

```
# C is cost matrix, shape of (nr_sample, nr_gt)
C = cost_class + cost_l1 + cost_giou

# Minimum cost, src_ind is index of positive sample
_, src_ind = torch.min(C, dim=0)
tgt_ind = torch.arange(nr_gt)
```


#### Notes
- [Review by 1st author on Zhihu](https://zhuanlan.zhihu.com/p/336016003)
- How does the pattern emerge? 
	- predicted location cost指的是使用預測結果和gt計算cost。在训练初期，网络的预测倾向于随机猜测，这样是否会将FP或者说并不是很好的正样本分配给了gt，从而影响模型的“成长”（上限），容易陷入局部最优？--> 这类方法其实都是利用网络当前阶段的能力，类似的还有前不久的jianfeng wang那篇，还有去年的FSAF那篇；像你说的其实确实有点问题，但实际上都挺work的
