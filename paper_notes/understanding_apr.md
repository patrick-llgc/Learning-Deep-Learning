# [Understanding the Limitations of CNN-based Absolute Camera Pose Regression](https://arxiv.org/abs/1903.07504)

_May 2020_

tl;dr: Pose regression algorithm is making interpolation between training data and is still far from being practically relevant.

#### Overall impression
Previous pose regression algorithms (camera relocalization) can be divided into two categories, APR (absolute pose regression) such as [PoseNet](posenet.md) and MapNet, and RPR (relative pose regression) such as RelocNet, Relative PN, **AnchorNet**. (AnchorNet come closest to structure based approaches). 

Structure based approaches (such as **Active Search**) is still the SOTA for camera relocalization. It uses Root-SIFT features to establish 2D-3D matches. 3D points in a scene and PnP algorithm within a RANSAC loop.

APR tries to map each image in the training set into a base translation, and predicts a linear combination (or conical combination)

APR tend to predict nearest neighbor to the test image in training set, if test set is far away from training set. Image retrieval method is closely related to APR method in that it computes a high dimensional embedding of the input image. The SOTA method for image retrieval is **DenseVLAD** (CVPR 2015). 

Camera relocalization is different from SLAM or SfM method that the 3D scene is known ahead of time. 

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- [Youtube demo](https://www.youtube.com/watch?v=7Efueln55P4)

