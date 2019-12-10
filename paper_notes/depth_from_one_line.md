# [Parse Geometry from a Line: Monocular Depth Estimation with Partial Laser Observation](https://arxiv.org/pdf/1611.02174.pdf)

_December 2019_

tl;dr: Predict dense depth from one-line lidar guided by RGB image. 

#### Overall impression
This paper proposes to "sculpt" the entire depth image from a reference while the original depth prediction task is to creating a depth value from the unknown. This makes the problem more tractable.

Monocular depth estimation is an ill-posed problem. Using sparse depth information can be helpful in solving the scale ambiguity. Refer to [Deep Depth Completion of a Single RGB-D Imag](deep_depth_completion_rgbd.md) and [deep lidar](deeplidar.md) and [sparse to dense](sparse_to_dense.md) for depth completion from unstructured sparse data.

#### Key ideas
- For each point in the imputed laser scan, generate a line along the gravity direction in 3D, then projecting back to 2D. --> generating a vertical line directly should largely yield the same results. 
- Add the reference depth map to the network output to predict depth. This means the network only has to learn the residual depth. 

#### Technical details
- Interpolation is used to fill in the blanks in the horizontal direction before populating in the vertical one. This is potentially dangerous as it introduces spurious data point in mid-air.
- Mixed classification and regression loss
	- multibin cls: the predicted value is with weighted average of all bins. 
	- Softmax loss: when prediction falls into the correct bin, cls loss vanishes. This can be extended to cross entropy loss used in [DC](depth_coeff.md).
	$$L_c = \sum_{i=1}^{M}\sum_{k=1}^{K} \delta([y_i] - k_i) \log(p^k_i) = \sum_{i=1}^{M} \log p^{[y_i]}$$
	- regression with L1 loss.
	- for improved regression, see [SMWA](smwa.md) or [DC](depth_coeff.md)

#### Notes
- This idea of using complementary sensor information can be extended to depth prediction using radar and rgb image.

