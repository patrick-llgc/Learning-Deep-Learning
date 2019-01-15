# [Panoptic Feature Pyramid Networks](https://arxiv.org/pdf/1901.02446.pdf)

_Jan 2019_

tl;dr: Add a semantic segmentation branch to Mask RCNN with FPN yields SOTA results in panoptic segmentation.

#### Overall impression
The idea is related to Retina-Unet, but the ablation test here is more solid. A single network can perform both semantic and instance segmentation. But if we are only interested in instance segmentation, the benefit from the addition of semantic segmentation is mininal. 

#### Key ideas
* The paper proposed a single network that simultaneously generates region-based outputs (for instance segmentation) and dense-pixel outputs (for semantic segmentation. The accuracy is equivalent to training two separate FPNs with R50 backbone, with essentially half the compute. With the same compute buget, then panoptic with R101 is strictly beneficial than two R50 FPNs. 
* FPN compared to FCN and U-Net
  * FPN is an alternative to FCN with dialted/atrous convolution. Dilated convolution has high computatation and memory footprint.
  * FPN is different from U-Net in that it uses a light weight decoder (asymmetric). It is ~2x more efficient than U-Net. 
* FPN now has two branches: one for instance segmentation/object detection and the other added on top of it for semantic segmentation.
  * FPN generates a pyramid (P2 to P5, 1/4 to 1/32) of scalesand each level with the same channel dimension (256). Each level makes independent object detection predictions which are later agggregated.
  * Semantic FPN branch upsamples each level of pyramid to the same resolution (1/4) and channel numbers (128), and sums the feature maps.
* The semantic segmentation loss is pixel-wise cross entropy loss. Simply adding them degrades the final performance for one of the tasks. This is corrected by tuning the relative weight of the two branches, and the optimal weight is quite different on different datasets (COCO vs Cityscape). 
* Semantic segmentation branch output a single class "other" for all object classes. However, training on and predicting all thing classes performs better (although the predictions are discarded).
* Adding a second task can help the main task with properly tuned loss weight.
  * Adding a semantic segmentation branch can slightly improve instance segmentation results over a single-task baseline .
  * Adding an instance segmentation branchcan provide even stronger benefits for semantic segmentation over a single-task baseline.

#### Notes
* The semantic FPN is built on top of the orignal instance segmentation FPN, and is not built on back-bone directly. (why not?)
* The improvement on object detection only benefited slightly from semantic segmentation, even with loss weight tuning. This is true more for COCO than Cityscape. Maybe the metric mAP is too stringent? What happens if we use AP10? No huge difference is observed in the improvement is observed between mAP and AP50 though.

