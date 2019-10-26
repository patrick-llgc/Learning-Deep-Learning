# [Deep Optics for Monocular Depth Estimation and 3D Object Detection](https://arxiv.org/abs/1904.08601)

_October 2019_

tl;dr: End-to-end design of optics and imaging process using coded defocus as additional depth cue.

#### Overall impression
This paper introduces the idea of adding a defocus blur and aberration without hurting the 2D performance. The idea of co-designing optics and image processing is core to computation photography.

That means using a special lens we can capture a carefully "blurred" image from which the depth information can be recovered easily for depth-dependent task such as monocular 3D object detection.

This paper proves the feasibility of an interesting idea, but it has a long way to go before industrial application. In addition, how this would impact the detection of small object is yet to be proved.

#### Key ideas
- Use all-in-focus image and GT depth map to simulate a "deliberately defocused" image. This process encode depth information into the blurred image. This image is then fed into a simple U-Net to extract depth information. 
- Meanwhile the 2D object detection task is shown to remain the same by this blurring process

#### Technical details
> We do not
claim to conclusively surpass existing methods, as we use
the ground truth or pseudo-truth depth map in simulating
our sensor images, and we are limited to an approximate,
discretized, layer-based image formation model.

> Lens optimized for depth estimation maintains
2D object detection performance while further improving
3D object detection from a single image

#### Notes
- The blurring process may hurt performance of detection of small object at a distance. The experiment on 2D detection is done on KITTI where most of the bbox is quite large and thus insensitive to this blurring.

