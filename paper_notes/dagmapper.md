# [DAGMapper: Learning to Map by Discovering Lane Topology](http://openaccess.thecvf.com/content_ICCV_2019/papers/Homayounfar_DAGMapper_Learning_to_Map_by_Discovering_Lane_Topology_ICCV_2019_paper.pdf)

_August 2020_

tl;dr: Use RNN to draw DAG boundaries of lane lines.

#### Overall impression
There are several works from Uber ATG that extracts polyline representation based on BEV maps.

- [Crosswalk Extractor](deep_structured_crosswalk.md)
- [Boundary Extractor](boundary_extractor.md)
- [Polyline Loss](hran.md): lane lines
- [DAGMapper](dagmapper.md): merges/forks

This is one application of RNN in boundary extraction. Previous work include [Polygon-RNN](http://www.cs.toronto.edu/polyrnn/poly_cvpr17/), [Polygon-RNN++](http://www.cs.toronto.edu/polyrnn/), [Curve GCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ling_Fast_Interactive_Object_Annotation_With_Curve-GCN_CVPR_2019_paper.pdf) also from Uber ATG. The main idea is to create a structured boundary to boost the efficiency for human-in-the-loop  annotation.

[Polyline Loss](hran.md) focuses on easier lane topology on highways, and [DAGMapper](dagmapper.md) focuses on highway driving, and focuses on hard cases like forks and merges. [Polymapper](polymapper.md) only focuses on extracting road network and do not have lane-level information. 

The tool is based on RNN, thus autoregressive and does not have a constant runtime for images with varying number of nodes. 

The way DAGMapper defines **node (control points)** and calculate their loss is very insightful. There is no unique way to define control points, and therefore instead of directly regressing L1/L2 distance of prediction and annotated control points, a Chamfer distance loss is used, which calculates the normalized distance between two densely sampled curves. 

#### Key ideas
- Loss: [Chamfer distance](http://vision.cs.utexas.edu/378h-fall2015/slides/lecture4.pdf).
	- Evaluated on densely sampled polyline points. 
	- **Adding or removing a control points in a straight line will not change loss.**
- Curve matching: Dilate each curve with a radius then compare IoU. This can be seen as a different way to compare two curves as compared to Chamfer distance.
- Given an initial point
	- Predict turning angle
	- Predict next node location
	- Predict status (merge, fork, continue)
- **DT (distance transformation)** is an efficient feature for mapping
	- Thresholded invert DT
	- Encodes at each point of the image the relative distance to the closest lane boundary. 
	- Threshold, binarize and skeletonize DT and use the endpoints as seeds. --> How?
- Results: 
	- P/R/F1 = 0.76 @ 2pix = 10 cm threshold. This is evaluated with the densely sampled polyline points. 
	- P/R/F1 = 0.96 @ 10 pix = 50 cm.

#### Technical details
- HD maps 
	- contain information about location of lanes, lane line types, crosswalks, traffic lights, rules at intersection, etc. 
	- HD map has cm level accuracy.
	- Semantic landmarks in HD maps are annotated by hand in an BEV image. 
- Resolution: 5 cm / pixel

#### Notes
- Many mapping papers before only focus on the coarse level of mapping (no lane-level information), such as [PolyMapper](polymapper.md), . They focus on road network extraction and semantic labeling, and are not suitable for autonomous driving.
- HD map + DL papers include 
	- [End-to-End Deep Structured Models for Drawing Crosswalks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Justin_Liang_End-to-End_Deep_Structured_ECCV_2018_paper.pdf) <kbd>ECCV 2018</kbd>
	- [Hierarchical Recurrent Attention Networks for Structured Online Maps](https://openaccess.thecvf.com/content_cvpr_2018/papers/Homayounfar_Hierarchical_Recurrent_Attention_CVPR_2018_paper.pdf) <kbd>CVPR 2018</kbd>
	- [Convolutional Recurrent Network for Road Boundary Extraction](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Convolutional_Recurrent_Network_for_Road_Boundary_Extraction_CVPR_2019_paper.pdf) <kbd>CVPR 2019</kbd>

