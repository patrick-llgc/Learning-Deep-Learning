# [Monocular 3D Object Detection in Cylindrical Images from Fisheye Cameras](https://arxiv.org/abs/2003.03759)

_December 2020_

tl;dr: Use cylindrical representation of fisheye images to transfer pinhole camera image to fisheye images. 

#### Overall impression
In pinhole camera model, perceived objects become smaller as they become further away as measured by depth Z. Objects with constant Z, regardless of X and Y positions, appear similar. Pinhole camera model can only accommodate limited FoV (view angles up to 90 deg).

Fisheye model, $r = f\theta$. It can represent view angles beyond 90 deg. In fisheye image, when an object moves in XY plane at the same Z, its appearance changes as well. CNN is not immediately compatible with such a fisheye raw image. 

One way is to convert it into a cylindrical view. An object's side and appearance remain the same as long as the $\rho$ distance keeps the same.

#### Key ideas 
- Change fisheye raw image to cylindrical view
- Interpret output z as $\rho$
- Use self-supervised learning (rotation prediction, etc) and finetune on small number (<100) of samples. The fintuning even on a small number of images helps a lot. 
	- We need finetuning as the analogy between cylindrical and perspective image (approximation that $\Delta X = \rho \Delta \phi$) breaks down for close-by objects. 

#### Technical details
- Warping uses predefined mappings and its computation time is typically negligible (as compared to model inference time).

#### Notes
- Questions and notes on how to improve/revise the current work  

