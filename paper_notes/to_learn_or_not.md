# [To Learn or Not to Learn: Visual Localization from Essential Matrices](https://arxiv.org/abs/1908.01293)

_May 2020_

tl;dr:  3D structure > SIFT + 5pt solver > Neural network based.

#### Overall impression
The paper builds on [Understanding APR](understanding_apr.md) and demonstrated that the good old SIFT + 5 pt solver is still the state of the art for relative pose estimation without 3D structures. 3D structures can achieve better results but require scene specific 3D modeling and lack generalization. 

Relative localization has three steps: feature extraction, find matching, and calculate essential matrices (or R and t).

- Feature extraction (SIFT or DL)
- Matching (concat or DL neighborhood consensus matching layer)
- Solve for pose (5 pt solver, DL regression)

The bottleneck of DL based approach is the matching and pose regression part. DL regression cannot generalize to new scenes as **DL cannot properly learn implicit matching by regression network**. 

However, even if we replace the pose regression with 5 pt solver, it still cannot beat SIFT + 5 pt solver. **This is mainly due to that current CNN features are coarsely localized on the image, that is, the features from the later layers are not mapped to a single pixel but rather an image patch.** All the self-supervised keypoints learner feature based methods still cannot beat SIFT consistently. I wrote a blog about self-supervised keypoint learning [here](https://towardsdatascience.com/self-supervised-keypoint-learning-aade18081fc3). As pointed out in an open review for KP2D, “the problem is old yet not fully solved yet, because handcrafted **SIFT** is still winning the benchmarks.”


#### Key ideas
- Direct approach:
	- Current SOTA of visual localization is based on 3D information. The representation are scene-specific and do not generalize to unseen scenes. 
- Indirect approach:
	- A more flexible way is relative pose estimation based on image retrieval first. This involves building a dataset based on posed images. It is more scalable and generalizable. The image retrieval can be done efficiently with compact image level descriptors (such as [Dense-VLAD](http://openaccess.thecvf.com/content_cvpr_2015/papers/Torii_247_Place_Recognition_2015_CVPR_paper.pdf) <kbd>CVPR 2015</kbd>). 
- Regressing R and t direclty needs finetuning hyperparameters to balance the two loss terms based on different scenes (outdoor vs indoors). Regressing essential matrix naturally handles this issue. 

#### Technical details
- Interpolating pose based on Nearest neighbor does not have the best results. The scenes have to be a min dist apart to avoid too much correlation. The paper starts with top ranked images and picks images within [3, 50] meters to all previous selected image for *outdoor* scenes. 
- Autonomous driving is largely planar motion and should be easier than a full blown 6 DoF localization.
- The essential matrix is regressed with a FC layer and may not be valid. The DLA regressed E matrix is then proejcted to a valid essential matrix space by replacing the first two singlular values by their mean and set the smallest singular value to 0. $\Sigma = diag(\sigma_1, \sigma_2, \sigma_3) \rightarrow diag(1, 1, 0)$. 
- Absolute pose estimation is scene specific as it depends on the coordinate system used. 

#### Notes
- 5 pts solver is non-linear. Easier way to do this is using an 8 pts solver which is linear. 

