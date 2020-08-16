# [Deep Structured Crosswalk: End-to-End Deep Structured Models for Drawing Crosswalks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Justin_Liang_End-to-End_Deep_Structured_ECCV_2018_paper.pdf)

_August 2020_

tl;dr: Extract structured crosswalk from BEV images.

#### Overall impression
There are several works from Uber ATG that extracts polyline representation based on BEV maps.

- [Crosswalk Extractor](deep_structured_crosswalk.md)
- [Boundary Extractor](boundary_extractor.md)
- [Polyline Loss](polyline_loss.md): lane lines
- [DAGMapper](dagmapper.md): merges/forks

This work predict deep feature maps, and use energy maximization to perform inference. Not exactly end to end. 

[Deep Structured Crosswalk](deep_structured_crosswalk.md) can be directly applied to extract road boundaries. [Deep Boundary Extractor](deep_boundary_extractor.md) is inspired by [Deep Structured Crosswalk](deep_structured_crosswalk.md) and uses conv-Snake to predict in an autoregressive fashion.

In a sense, this work is basically **edge-aware semantic segmentation**. The structured prediction module converts the unstructured semantic segmentation results into a structured presentation.

#### Key ideas
- Input: lidar + BEV cam, 4 ch. CenterLine from OSM (OpenStreetMap) needed for prediction and dataset generation.
- Output (deep features):
	- Inverse distance transform (DT, 1 ch)
	- Semantic segmentation (1 ch)
	- Predicted Alignment angle (Dilated normals, 2 ch) --> This is improved to direction map in [Deep Boundary Extractor](deep_boundary_extractor.md).
- Inference with **Structured prediction module**.
	- finds the best two boundaries x1 and x2 along with the best angle β by maximizing a structured energy function.
	- Draw two orientated and parallel lines so that it tightly encloses the maximum number of segmented points 
	- It also encourages draw the lines along the DT boundary.

#### Technical details
- 96% accuracy.
- 4 cm/pixel resolution
- Distance transforms a natural way to “blur” feature locations geometrically ([source](https://www.cs.cornell.edu/courses/cs664/2008sp/handouts/cs664-7-dtrans.pdf)). This is another way to densify sparse GT as compared to Gaussian blurring as in [CenterNet](centernet.md).
- Lidar helps, and multiple passes helps. Multiple BEV is better than one pass BEV + lidar.
- Ablation study
	- **Alignment angle** helps quite a bit and without it there is significant drop in performance. Also this is predicted quite accurately as substitute the predicted angle with oracle (GT) angle does not lead to performance increase. 
	- **Semantic segmentation alone is a strong baseline**, and already achieves quite good results.
- GT Noise: They compare the noise in human annotation of the ground truth by annotating 100 intersections with several annotators. About 5% error in IoU.
- Uses integral accumulators (images), calculations can be optimized to avoid exhaustive evaluation.

#### Notes
- What is the image size (HxW) of each patch?
- This crosswalk extraction task can be easily extended to road boundary extraction and stopline extraction. 