# [FoveaBox: Beyond Anchor-based Object Detector](https://arxiv.org/pdf/1904.03797v1.pdf)

_April 2019_

tl;dr: RetinaNet based anchor-free method to predict center and top-left/bottom-right corners. It is more robust in detecting long objects. 

#### Overall impression
This paper is one of the many recent papers about anchor-free object detection. FoveaBox and [FSAF](fsaf_detection.md) both build upon RetinaNet, but FSAF selects feature level by learning, but FoveaBox uses overlapping levels to detect an object. FoveaBox can also be used with anchor-based RetinaNet results, which performs better than FSAF.

#### Key ideas
- Anchor box can be regarded as a feature-sharing sliding window scheme.
- We human naturally recognize the object in the visual scene without enumerating the candidate boxes. For human eyes, the center of the vision field is with the highest visual acuity.
- In RetinaNet, each cls network predicts KA numbers (A=9, number of anchor boxes), and each reg network predicts 4A numbers at each bin of the feature map. In contrast, FoveaBox predicts K and 4 numbers for each feature map bin. This is very similar to FSAF.
- It keeps the scale assignment to different levels of FPN to stabilize training. This is a key difference to YOLOv1. This is very similar to an anchor-box scheme. In a way, FoveaBox still relies on some priors. 
	- Cls: A donut shaped exclusion area with shrunk factors of 0.3 and 0.4 of the diameter is used to separate the positive samples and negative samples.
	- Reg: still needs to scale offset to around 1 for faster and more stabilized training.
- Ablation:
	- Increasing beyond 6-9 anchors does not leads to improvement. 
	- Having overlap in scale assignment helps. Two adjacent levels are responsible for the same object. 
	- Anchor free method is more robust as it is not adapted to the anchor box statistics. 
	- FoveaBox can be used to replace FPN. 


#### Technical details
- We could use SoftNMS and bbox voting for post-processing.
- FoveaBox can be used to replace FPN. 
- FoveaBox can be used with anchor-based RetinaNet. 
- 42.1 AP with ResNeXt-101, but this is single scale. The authors did not report multi-scale test results. 

#### Notes
- Another concise description of anchor boxes in object detection.

	> Anchor method suggests dividing the box space (including position, scale, aspect ratio) into discrete bins and refining the object box in the corresponding bin.

