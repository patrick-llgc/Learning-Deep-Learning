# [Deep Metadata Fusion for Traffic Light to Lane Assignment](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8613841)

_August 2019_

tl;dr: Convert metadata to binary masks and use deep metadata fusion to improve traffic light to lane assignment.

#### Overall impression
This work improves upon the previous [only-vision baseline](https://ieeexplore.ieee.org/document/8569421) (from same authors, without any meta data).

The traffic lights to lane assignment problem can be solved traditionally using huristics rule-based methods, but not very reliably. Only-vision approach (enhanced with enlarged ROIs), and with meta data, works almost as good as human performance in subjective test. This is because the relevant traffic lights are not always mounted directly above the their associated lane.

**I feel that lane localization is very challenging.** How do we know if the current lane is right turn only? Are we going to rely on HD maps or vision alone for this? At least some vision based methods should be used. 

#### Key ideas
- Rule based methods achieves about 60%-80% precision in different (complex/full) driving scenarios. Rule-based methods largely works in non-complex cross-sections
	- largest and nearest traffic light in driving direction is relevant
	- it has to be the traffic light with the largest size and highest position from the group with most traffic lights of the same color
	- a laterally mounted traffic light between the left and right ego-vehicle lane line markings is relevant. If no traffic light is within the ego-vehicle lane line markings, the traffic light with the shortest lateral distance tot he ego-vehicle lane center will be selected as relevant.
- Architecture:
	- Input 256x256x3 + metadata feature map (MFM)
	- Output: 256 x 3 for ego, left and right lanes, indicating if the x coordinates are relevant. 

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

