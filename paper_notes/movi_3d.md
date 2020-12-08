# [MoVi-3D: Towards Generalization Across Depth for Monocular 3D Object Detection](https://arxiv.org/abs/1912.08035)

_January 2020_

tl;dr: Generate synthetic views (virtual cam/viewport) of image to reduce the complexity of 3D MOD neural networks.

#### Overall impression
The paper builds on the work of [monoDIS](monodis.md). The main idea is that the network has to build **distinct representations** devoted to recognize objects at specific depths and there is little margin of generalization for different depth. This happens as it lacks generalization across depth. As a result, we have to scale up network's capacity as a function of the depth ranges, and scale up training data as well. 

The paper basically proposes a patch-based distance prediction network so that the network only has to learn representation for distance/scale of a very limited range. 

This is a classical tradeoff of model/data complexity vs inference complexity. If there is an inherent structure of the image (in autonomous driving camera images, closer object appear at the bottom of the image and further away object are higher up in the image), it can be exploited using row-aware or **depth aware convolution** (cf [M3D RPN](m3d_rpn.md)). In this paper, they did a row-wise image pyramid of the original image.

![](https://cdn-images-1.medium.com/max/1440/1*pASNAVJhmkoZRHe37r6qPA.png)

The paper also has a good introduction of monocular 3d object detection.

#### Key ideas
- Training and inference discrepancy
- Training: train a NN to make correct predictions within a limited depth range.
	- generate nv = 8 **viewport** per original image. 
	- Ground truth guided sampling procedure (cf [PointRend](pointrend.md)). The object should be completely visible (not cropped). Random shift of viewport by [-Z_res/2, 0]. Z_res = 5 m. 
	- GT falling out of preset depth range [0, Z_res] is set to ignore/dont_care.
	- The depth is shifted by Zv to ensure depth invariance.
- Inference:
	- Sample every Z_res/2 (cut out horizontal strips). This IPM is based on accurate extrinsics. 
	- Adjust height to be the same
	

#### Technical details
- The paper did detailed analysis of the viewport intrinsic prarameter but they did not use it for training nor inference. Basically crop and rescale.
- With the patch-based strategy, class-balanced sampling can also be performed. 

#### Notes
- Q: can we use IPM to inference the distance for the close-by cars and then use viewport to generalize it to further distances? This bootstrap sounds too good to be true. 

