# [VectorMapNet: End-to-end Vectorized HD Map Learning](https://arxiv.org/abs/2206.08920)

_July 2022_

tl;dr: BEV perception of road layout with directly vectorized output.

#### Overall impression
Map elements nee a compact representation (vectorized representation) to ensure that they can be used for downstream tasks like prediction and planning.

This paper proposes the top-down approach of road layout prediction. This is quite different from most previous bottom-up approaches (such as [HDMapNet](hdmapnet.md) from the same group). Similar top-down approaches include [STSU](stsu.md), but the performance of VectorMapNet is significantly better and reaches new SOTA on BEV perception of road layout.

Overall the paper divides the detection of geometrically accurate map element into **two steps**, first detecting the map element in the BEV space and then predicting the local geometric details inside each element. Actually this gives strong incentives for future researchers to directly predict the lane line segments (map elements as referred to in this paper) in a centerNet-like one-stage detector (detecting the instance center and then offsets).

The paper is full of great insights. It casts new possibilities into the field of BEV perception of road layout. However the writing of the paper is a bit hard to follow, especially the two-stage detection method. The math notation is quite sloppy, especially in Session 2.2. Lots of reused letters which have different meanings in fact. The ablation study could have been more thorough. For example, the design choices in Figure 3 was not ablated (see Notes for details).


#### Key ideas
- Elements in the map are modeled as **polyline primitives**. Every map element is an indispensable part of the traffic rule that restrains every traffic participant. Using polyline as primitive allows the method to model maps without complicated geometric assumptions and express map element with arbitrary shapes.
- The decoder architecture has two DETR chained in series.
	- Stage 1 DETR: Map element detection
		- In this detection task, not only one keypoint per element is detected, à la CenterNet. Multiple keypoints are detected. 
		- Keypoint representation: Note that the keypoints do not need to fall on the final polyline. Ways to represent the element include bbox, extreme points and Start-mid-end.
		- The DETR predicts $N_{max}$ * k points. $N_{max}$ elements, and each element has k points. Each keypoint also has two embeddings, indicating which element it belongs to, and its ordinal position in that element.
	- Stage 2 DETR: Polyline generation, based on the results from Stage 1.
		- Polyline embedding: 3 embeddings, indicating whether it is x or y coordinate, which vertex the token belongs to (why is this needed?), and value embedding (discrete bins, instead of continuous regression values).
		- The paper observed that the discrete values are better to predict than continuous values. This idea is similar to [DORN](dorn.md).
- Evaluation with [Chamfer distance](hran.md) or Fréchet distance.
	- Chamfer distance: average of minimum distance between all points pairs. (平均最短距离)
	- Fréchet distance: maximum of minimum distance between all points pairs. (最大最短距离)
	- Not sure why there are 3 thresholds for Chamfer distance while there is only 1 threshold for Fréchet distance.
- Curve sampling strategies
	- Curvature based sampling: sample when curvature is larger than a predefiend threashold.
	- Fixed sampling: every 1 meter.
	- Curvature based is better than fixed sampling by a large margin. The authors reason that the redundant vertices hurt the learning by averaging out the weights of essential vertices.

#### Technical details
- The scope of prediction is the same as [HDMapNet](hdmapnet.md) and many other studies: lane dividers (lane lines), road boundaries, pedestrian crossing.
- BEV transformation is simple IPM, with four different ground height assumption (-1, 0, 1, 2m). This is simple and effective as a poor man's BEV feature transformation, and can be improved by other more powerful alternatives such as [BEVFormer](bevformer.md) or [PersFormer](persformer.md).


#### Notes
- Questions
	- Which design of keypoint representation of map element in Fig 5 was used in the best model?
	- Was the polyline generator autoregressive or with simultaneous output like in DETR? There are conflicting information in the paper. (Fig. 2 suggests it is DETR like, but Session 2.3 indicates it is autoregressive).
	- How were the polylines divided (multiple equivalent ways exist) were not discussed.
- I should perhaps write a review on polyline learning. Papers to read include:
	- [LETR: Line Segment Detection Using Transformers without Edges](https://arxiv.org/abs/2101.01909) <kbd>CVPR 2021 oral</kbd>
	- [HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps](https://openaccess.thecvf.com/content/CVPR2021/papers/Mi_HDMapGen_A_Hierarchical_Graph_Generative_Model_of_High_Definition_Maps_CVPR_2021_paper.pdf) <kbd>CVPR 2021</kbd> [HD mapping]
	- [SketchRNN: A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477) [David Ha]
	- [PolyGen: An Autoregressive Generative Model of 3D Meshes](https://arxiv.org/abs/2002.10880) <kbd>ICML 2020</kbd>
