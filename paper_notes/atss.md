# [ATSS: Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424)

_May 2020_

tl;dr: The gap between anchor-based and anchor-free methods lie in the sampling strategy.

#### Overall impression
The paper founds out that the main difference between anchor-based methods (such as RetinaNet) and anchor-free methods ([FCOS](fcos.md)) mainly lies in the definition of positive examples and negative examples.

This paper draws much inspiration with [FCOS](fcos.md) and shall be read together. For anchor based networks With ATSS, one bbox

#### Key ideas
- The paper first uses the same tricks for both RetinaNet and FCOS to make sure these training tricks are accounted for during comparison
	- [GroupNorm](groupnorm.md) in heads
	- [GIoU loss](giou.md)
	- In GT bbox: positive samples should be inside GT bbox
	- Add a trainable scaler for each FPN level
	- Introducing centerness branch
- Selection of positive samples and negative samples
	- RetinaNet uses IoU to select positive anchor bbox
	- FCOS uses spatial constraint (anchor point should be inside GT) and scale constraints (uses predefined ranges to assign GT to different levels of FPN)
	- ATSS: 
		- from each level L select k anchors (kL in total), whose centers are closest to GT bbox
		- Compute IoU and mean and std (for anchor-free method, the anchor point is converted to anchor box with 8S scale)
		- Select bbox with IoU > mean + std and center of bbox in GT (usually 20% * kL
- Regressing from **a point or a preset anchor box** yields the same results.
- For IoU based sampling, more anchor box is helpful. For ATSS, one bbox is enough.

#### Technical details
- The authors reimplemented many existing work to make side by side comparison. This is a solid way to dive into the details of a system, and pick apart what works and what not.

#### Notes
- Questions and notes on how to improve/revise the current work  

