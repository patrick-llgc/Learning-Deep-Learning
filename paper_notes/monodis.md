# [MonoDIS: Disentangling Monocular 3D Object Detection](https://arxiv.org/abs/1905.12365) 

_August 2019_

tl;dr: end2end training of 2D and 3D heads on top of RetinaNet for monocular 3D object detection.

#### Overall impression
The paper is articulate in specifying the input and output of the network, and info required at training and inference. It does not require depth information, only the RGB and the 3D bbox info.

The paper proposes a **disentangling transformation** to split the original combinational loss (e.g., size and location of bbox at the same time) into different groups, each group only contains the loss of one group of parameters and the rest using the GT. Note that sometimes the loss is already disentangled, such as those originally proposed by YOLO or Faster RCNN. This only applies to losses with complicated transformation such as 3d bbox corner loss and sIOU loss as proposed in this paper. 

#### Key ideas
- Architecture: RetinaNet (FPN + Focal loss) + 2D sIoU loss + 3D corner loss.
- Disentangling transformation of loss. This is only relevant when loss terms are entangles with multiple output of the network.
- The original 11-point AP is not reasonable as it includes a precision value at recall=0. This is valid, but perhaps not that important as AP_11 and the proposed AP_40 have the same trend.

#### Technical details
- In place ABN to replace BatchNorm + ReLU.
- Postprocess for 2D obeject detection: filter by thresh + topk --> NMS --> topk.
- Signed IoU loss to prevent vanishing gradient. If two bbox are disjoint, then sIoU is negative. Overall sIoU is in [-1, 1].
- 3D corner loss: but with 10-dim of vectors all entangled. --> Use disentanglement transformation. The quaternion regression also only regresses the observation angle, or allocentric angle, similar to [deep3dbox](deep3dbox.md) and [monoPSR](monopsr.md).
- conf = 2d conf * (3D conf | 2D) for inference filtering.
- The paper has a nice summary of KPIs used in detection track. In particular, nuscense dataset has a overall combined metric called NDS (see [github implementation](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/detection)).

#### Notes
- The method has relatively relaxed data requirement for training, but still requires 3D bbox annotatation.
- We could also use the sIoU loss for 2D object detection.

