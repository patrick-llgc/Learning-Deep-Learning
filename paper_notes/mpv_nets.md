# [MPV-Nets: Monocular Plan View Networks for Autonomous Driving](https://arxiv.org/abs/1905.06937)

_September 2020_

tl;dr: Project 3D object detection into BEV map to train a better driving agent.

#### Overall impression
Monocular 3D object detection in a way similar to [Deep3DBox](deep3dbox.md). Then the 3D object detection results are rendered into a BEV (Plan view). Having access to this plan view reduces collisions by half.

#### Key ideas
- Plan view is essential for planning. 
	- In perspective view, free space and overall structure is implicit rather than explicit.
	- Hallucinating a top-down view of the road makes it easier to earn to drive as free and occupied spaces are explicitly represented at a constant resolution through the image.
	- Perception stack should generate this plan view for planning stack.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

