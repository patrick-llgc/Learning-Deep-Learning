# [KP3D: Self-Supervised 3D Keypoint Learning for Ego-motion Estimation](https://arxiv.org/abs/1912.03426)

_March 2020_

tl;dr: Predict keypoints and depth from videos simultaneously and in a unsupervised fashion.

#### Overall impression
This paper is based on two streams of unsupervised research based on video. The first is depth estimation starting from [sfm Learner](sfm_learner.md), [depth in the wild](learnk.md) and [scale-consistent sfm Learner](sc_sfm_learner.md), and the second is the self-supervised keypoint learning starting from [superpoint](superpoint.md), [unsuperpoint](unsuperpoint.md) and [unsuperpoint with outlier rejection](kp2d.md).

The two major enablers of this research is [scale-consistent sfm Learner](sc_sfm_learner.md) and [unsuperpoint](unsuperpoint.md).

The main idea seems to be using sparse matched keypoint pairs to perform more accurate (relative) pose estimation. Previously the ego motion is directly regressed from two stacked neighboring images. This leads to much better ego motion estimation.

Both [KP3D](kp3d.md) and [D3VO](d3vo.md) uses DSO as backned, and KP3D reaches on par performance with DVSO while D3VO beats DVSO.

#### Key ideas
- Some notation convention:
	- $p_t \in I_t$ keypoints in target image and $p_s \in I_s$ keypoints in source image
	- $p_t^{MV} \in I_s$ matched keypoints of $p_t$ in source image based on descriptor space. Based on the pair of $p_t \leftrightarrow p_t^{MV}$ we can compute the associated ego motion $x_{t \rightarrow s}$. Descriptor loss is based on this. 
	- $p_t^* \in I_s$ warped $p_t$ in source image ($\hat{p_t}$ in KP2D ). Sparse keypoint location loss is between $p_t^{MV}$ and $p_t^*$.
	- Once $x_{t \rightarrow s}$ is known, dense Photometric loss and sparse keypoint location loss are formulated.
- In the whole pipeline, calculating $p_t^*$ is the hardest. In Homography Adaptation $p_t^*$ can be calculated trivially, but in multi-view adaptation this is hard and need to project to 3D via $\pi^{-1}(R|t)$.
- Instead of using CNN directly for pose estimation (PoseNet in [sfm Learner](sfm_learner.md)), KP3D uses matched keypoint to do pose estimation, and this could be the key to the better performance ([superpoint](superpoint.md) and [unsuperpoint](unsuperpoint.md) are known to yield very good HA, homography accuracy).
- Added depth consistency, as the depth is scale-ambiguous. It is critical for ego-motion estimation. A sparse loss between $p_t$ and $p_t^{MV}$ is used. 
- Pose estimation from matched 2D points:
	- Conventional method uses epipolar geometry constraint to get R and t
	- As we have estimated Depth, we can do Epnp. However this is not differentiable and thus can only be used as an initial guess
	- We can roughly use transformed 3d position (with initial guess) of keypoints to get 3D location of points in new camera coordinate. Then use the SVD based method ([Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm) to get a closed form solution. This is one special case of the ICP algorithm. 

#### Technical details
- Training process:
	- Pretraining keypoint detector (similar to [KP2D](kp2d.md)).
	- KeypointNet and DepthNet both imageNet pretrained ResNet18. 
	- Changing backbone from VGG in [KP2D](kp2d) to ResNet18 in [KP3D](kp3d.md) improves performance.

#### Notes
- Both Kabsch algorithm and ICP is solution to the problem of [Orthogonal Procrustes problem 正交普鲁克问题](https://en.wikipedia.org/wiki/Iterative_closest_point). 
- 普洛克路斯忒斯 (Procrustes) 是希臘神話中海神波塞頓 (Poseidon) 的兒子。他在雅典到埃萊夫西納 (Eleusis) 的神聖之路 (The Sacred Way) 上開設一間黑店，向路過的旅人謊稱店內設有一張適合所有人的鐵床。旅客投宿時，普洛克路斯忒斯將身高者截斷雙足，身矮者則強行拉長，使之與床的長短相同。從來沒有一個人的身長與鐵床的長度相同而免於凌遲，因為他暗地裡準備了兩張床[1]。後人於是以 Procrustean 表示「削足適履，殺頭便冠」，意思是將不同的長度、大小或屬性安裝到一個任意的標準。
- 正交 Procrustes 问题：给定两个mxn阶实矩阵A和B，求一个nxn阶实正交矩阵$Q^T=Q^{-1}$使得  $|B-AQ|_F$ 具有最小值，其中F是 Frobenius 范数。
正交 Procrustes 问题是一个最小平方矩阵近似问题，可以这麽解读：A 是旅人，B 是铁床，正交 Procrustean 变换 (包含旋转和镜射) Q 即为施予旅人的肢体酷刑。我们的问题是求出一酷刑使旅人变形后的身长与铁床的长度最为吻合。


