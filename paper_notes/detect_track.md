# [Detect to Track and Track to Detect](https://arxiv.org/abs/1710.03958)

_January 2020_

tl;dr: Predict shift of bbox across frames for better tracking.

#### Overall impression
Conventionally, each frame is pushed through an object detector to get a list of bbox, independently. Then the list of bbox goes through a data association (Hungarian algorithm, e.g.) to form tracklets. The Hungarian matching usually takes the IoU as the matching criterion over frames.

This paper mainly addresses issues when there are large shifts from videos (global shifts) and when IoU across frames are not reliable anymore. It predicts the movement of each bbox from one frame to the next, from the correlation maps between frames. 

Using correlation for tracking stems from traditional CV. Correlation tracker.

The tracking method here is offline. According to [知乎](https://zhuanlan.zhihu.com/p/34633035):
> 关于offline的数据关联方法有很多，例如Max Flow Mini Cut、k-partite graph、multicut、crf、mrf等，主要是涉及组合、图论或者是概率图模型中的一些方法，而online算法目前以个人知识面只知道匈牙利算法Hungarian Algorithm（论文deepsort）和把multi-tracking当作强化学习RL的马尔科夫决策过程MDP过程(论文MDP_tracking)。

#### Key ideas
- Tracking loss: regress the movement of bbox for one bbox from one frame to the next. The input is stacked RoIPooled features from two frames and the (2d+1)**2 correlation maps.
- Linking tracklets to object tubes
$$s_{t\rightarrow t+\tau} = p_t + p_{t+\tau} + \psi (D_t, D_{t+\tau}, T_{t\rightarrow t+\tau})$$
- The optimal path can be solved efficiently by applying Viterbi algorithm.

#### Technical details
- The correlation maps takes in feature maps as input. It is translational equivariant.
- Correlation maps is calculated with a maximum displacement d, and has $H \times W \times (2d+1)*(2d+1)$ size.
- It takes in two images at the same time.
- The ROI used for tracking loss is from frame t

#### Notes
- Can we replace the correlation map with optical flow?
- Deep Feature Flow: calculate on key frames only and propagate features by optical flow.
- GOTURN (Learning to Track at 100 FPS with Deep Regression Networks): single target tracking
- Question: how to form pairs of roipool'ed features?
