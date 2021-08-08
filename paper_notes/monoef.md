# [Monocular 3D Object Detection: An Extrinsic Parameter Free Approach](https://arxiv.org/abs/2106.15796)

_August 2021_

tl;dr: Regresses the extrinsics and uses feature transfer to compensate.

#### Overall impression
This approach correctly addresses one drawbacks from existing mono3D dataset which assumes a fixed extrinsics. This is not true in industry applications due to potholed and uneven roads. The paper introduces a method to relax the constraint of assuming a fixed extrinsics. 

Personally I am not super confident about the approach to regress extrinsics with horizon and vanishing point. This may work on highways but in crowded urban scenario this may fail miserably. 

#### Key ideas
- Camera extrinsics regression with detecting vanishing point and horizon change
- Feature transfer by extrinsic parameters
	- Intuition: low-level features like edges are closely related to extrinsics (contents), while high-level features like texture and illumination are not related to extrinsics (style). This resembles the image style transfer method.
	- Two input images, one original and one disturbed. The disturbed image is restored/aligned by the predicted extrinsics.
	- Content loss: between the feature map of the original image and the realigned disturbed image.
	- Style loss (Frobenius norm of Gram matrix)

#### Technical details
- Summary of technical details

#### Notes
- The mathematical formulation is not easy to follow and I am not sure I fully understand. I may need to revisit one day if necessary.
- We perhaps can do data augmentation to boost the robustness against extrinsics change.
- [Code on github](https://github.com/ZhouYunsong-SJTU/MonoEF) to be released as of 08/08/2021.

