# [SKU110K: Precise Detection in Densely Packed Scenes](https://arxiv.org/abs/1904.00853)

_October 2020_

tl;dr: object detection in crowded (but not necessarily occluded scene), and proposed one new retail dataset SKU110k.

#### Overall impression
This paper proposes a novel method to do object detection in densely packed scene with a new NMS method. This method will not necessarily work when there is a lot of occlusion. Fortunately for retail scenes, occlusion does not seem to be a huge problem. 

The added IoU branch looks much like that in [IoU Net](iou_net.md). In both methods, the IoU score (instead of the classification score) is used for bbox NMS. However [SKU110k](sku110k.md) also 

This problem may be much better tackled with segmentation or keypoint detection method. In other words, anchor free methods like [CenterNet](centernet.md) or [FCOS](fcos.md) may work perfectly in this scenario. 

#### Key ideas
- Each bbox has an IoU score prediction. It is trained with a binary CE loss.
	- A good objectness cls would be invariant to occlusion and translation.
- EM Merger as an alternative to NMS. It filters, merges the overlapping detection clusters before NMS to resolve a single object per object. 
	- Detections as Gaussians. Convert each detection to a Gaussian blob with std proportional to the size of the bbox. This generates a heatmap from weighted average of Gaussian blobs. --> this is much like [probabilistic object detection](pdq.md).
	- Mixture of Gaussian (MoG) clustering method. The clustering is solved with EM method. The bbox generated Gaussian blobs have diagonal covariances.
	- Gaussians as detections. Search pre-NMS bbox with centers in the ellipse (isometric curve with 2 sigmas). Take the median dimensions of the detections in this set to recover final detection bbox. 

#### Technical details
- SKU110K dataset has ~150 images per image. 
- In densely packed scenes, multiple overlapping bbox often reflect multiple, tightly packed objects, many of which receive high objectness scores. 
- NMS is a **hand-crafted** algorithm applied at test time as post-processing. GossipNet tries to learn this in a new layer in the neural network.

#### Notes


