# [SGDepth: Self-Supervised Monocular Depth Estimation: Solving the Dynamic Object Problem by Semantic Guidance](https://arxiv.org/abs/2007.06936)

_August 2020_

tl;dr: Build a [Mannequin dataset](frozen_depth.md) for monodepth. Use segmentation mask to filter out real moving object.

#### Overall impression
The paper addresses the moving object issue by adaptively filter out regions that has large dynamic movement. 

Segmentation techniques are also used in [Every Pixel Counts](every_pixel_counts.md) which proposes an implicit binary segmentation. [SGDepth](sgdepth.md) does not extend the image projection model to include cars, but simply exclude the car pixels. But this alone will lead to poor performance as depth of car pixels will not be learned at all.

But this method still seems to suffer from the **infinite depth problem**. We need to integrate the depth estimation with [depth hints](depth_hints.md). [PackNet-SG](packnet_sg.md) provides an intuitive way to 

[SGDepth](sgdepth.md) develops a method to detect frames with non-moving cars, similar to that of [Mannequin dataset](frozen_depth.md). In other words, moving cars should be excluded from loss computation while stationary cars should still be used. 


#### Key ideas
- Major problems with monodepth
	- Occlusion/disocclusion
	- Static frames (little ego motion)
	- DC objects (Dynamic Class objects, cars/pedestrians/etc)
	- [Monodepth2](monodepth2.md) tackles the first two by minimum reprojection loss and automasking. Most previous projects left the third issue open.
- Loss
	- Min reproj loss
	- Smoothness loss
- **Warping mask and Masking out cars**
	- Like warping input images, but uses **nearest neighbor sampling** as the pixel values in semantic segmentation results do not have ordinal meaning.
	- If the warped mask and the predicted mask on the target image has large IoU, then we can assume that the cars are non-moving in the scene, and use it to train. Otherwise we would need to filter out all cars in the scene. 
	- **Scheduling** of masking thresholds: The threshold if dynamically determined by the fraction of images to be filtered out. In training, more and more images are not-masked out. Masking only guides training in the beginning, and the network sees more and more noisy samples. 

#### Technical details
- ENet for real time segmentation network
- Uses the same network (encoder + task specific decoder) to do both monodepth and semantic segmentation. This is different from the work in [Towards Scene Understanding: Unsupervised Monocular Depth Estimation with Semantic-aware Representation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Towards_Scene_Understanding_Unsupervised_Monocular_Depth_Estimation_With_Semantic-Aware_Representation_CVPR_2019_paper.pdf).
- Depth map prediction by predicting $1/(a \sigma + b)$, where $\sigma$ is the post-sigmoid prediction.

#### Notes
- [code on github](https://github.com/ifnspaml/SGDepth)
- Can we use optical flow and epipolar constraints to do motion segmentation? 
- If we do motion segmentation, then we can also tell if a car is parked or not. 
