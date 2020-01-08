# [ApolloCar3D: A Large 3D Car Instance Understanding Benchmark for Autonomous Driving](https://arxiv.org/abs/1811.12222)

_January 2020_

tl;dr: Detecting 3d pose of a car by detecting keypoints and using Epnp.

#### Overall impression
ApolloCar3D is a new dataset on ApolloScape. It has ~5k images and ~60k car instances. Around 10 cars per image. They filtered out images with fewer cars from ApolloScape. Compared with Kitti, this dataset has many cars that are distant. It annotates 66 keypoints. (but they also show that the many numbers of keypoints and their semantics are one major source of error in annotation).

Using keypoints significantly improves performance. 

#### Key ideas
- Keypoints are detected, and then fitted to 3D models with EPnP algorithms. 
- Co-planar constraints between cars and its neighboring cars. This only takes effect if less than 6 points are annotated. Number of neighboring cars is set to 2.
- Direct approach (top-down): 
	- regress amodal bbox and 3d center's 2d projection
	- 3D RCNN uses render and compare to find depth, and here they directly regresses depth using ODIN method. 
- Keypoint based method:
	- CPM (conv pose machine) to locate keypoints, based on 2D patches

#### Technical details
- Understanding 3D properties of objects from monocular image is called "inverse graphics". 
- Human performance study shows on average ~2 to 3 pixels difference.
- The shape models are highly accurate: on average the reprojected mesh and the mask contour is less than 3 pix.
- The main inaccuracy is the depth prediction for the direct method. Directly predicting global depth info from appearance is inaccurate.

#### Notes
- [EPnP algorithm in openCV](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
- The relative threshold is from 0.01 to 0.10, matching that of [Kaggle Competition](https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation)

