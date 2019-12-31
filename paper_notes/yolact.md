# [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)

_December 2019_

tl;dr: First real-time instance segmentation, by linearly combining activation maps and crop with bbox. 

#### Overall impression
This is a well written paper with great idea, and really solid engineering work. 

Most previous works (Mask RCNN) depends on feature localization (feature repooling) and repool the features to predict a fixed-resolution (14x14 or 28x28) mask. This is inherently sequential and hard to speed up.


#### Key ideas
- YOLACT breaks instance segmentation into two parallel tasks: 
	- 1) Generating a set of prototype masks --> **using FCN which are good at producing spatially coherent masks**
	- 2) Predicting per-instance mask coefficients --> **using fc to producing semantic vectors**
	- The assembly step is a simple linear combination realized by matrix multiplication
- The prototype masks are independent of categories. It learns a distributed representation and each instance mask is a linear combination of the prototypes. Prototype masks are image dependent and do not depend on any specific instance. 
- The emerging behavior of prototypes is interesting! (Fig. 5): some is position sensitive, and some detect contours.
- Advantages to Mask RCNN
	- Fast. The entire mask branch takes only ~5 ms!
	- Temporally stable. Single stage. Segment whole image then crop.
	- Better quality for larger object. No fixed size mask prediction
- Disadvantages to Mask RCNN
	- Worse overall performance. --> mainly in detection quality. In high threshold, it is even better than Mask RCNN.
	- YOLACT may leads to leakage, if bbox is not accurate. 

#### Technical details
- Predict prototype masks in 1/8 scale P3 (finest scale in FPN) and upscale to 1/4.
- Use ReLU for unbounded activations.
- Predict c+4+k coefficients. tanh after k coefficients to enable subtraction. 
- Mask assembly: $M = \sigma(PC^T)$, P is hxwxk and C is nxk. n is number of instances. The masks are probabilistic, thus using a sigmoid.
- ResNet is not exactly translation variant because of padding.
- k = 32 is the best. Adding more prototypes usually adds duplicates, and makes learning coefficients harder.
- Fast NMS performs NMS in parallel, allowing those otherwise be removed bbox to suppress lower bbox scores as well. This hurts 0.3 mAP.
- The performance gap between YOLACt and mask RCNN is 

#### Notes
- [GitHub page](https://github.com/dbolya/yolact) 

