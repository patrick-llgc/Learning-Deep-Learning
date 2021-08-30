# [SimTrack: Exploring Simple 3D Multi-Object Tracking for Autonomous Driving](https://arxiv.org/abs/2108.10312)

_August 2021_

tl;dr: Simplified tracking mechanism from [CenterPoint](centerpoint.md).

#### Overall impression
SimTrack stands for simplified tracking. This paper is heavily inspired by [CenterPoint](centerpoint.md). Instead of using the predicted motion for better matching (data association) in tracking, [SimTrack](simtrack.md) simplified that matching to a simple look-up (or read-off as in the paper).

The usage of combined map during inference time resembles **a simple version of convLSTM (memory)**. This helps the network to look beyond two immediate frames and thus handle occlusion better. It is not trained online specifically but only leveraged in inference time.

The key design is to predict the first-appear location of each object in a given snippet to get the tracking identity and then update the location based on motion estimation.

Difference with [CenterPoint](centerpoint.md):

- [CenterPoint](centerpoint.md) treat the forecasted detection as a bridge for object mat hing instead of using them as the final tracking output. 
- [CenterPoint](centerpoint.md) only cares about the location relationship objects between frames, but not their confidence. [CenterPoint](centerpoint.md) introduces the confidence of objects between frames via pooling features from previous frames.

![](https://pic3.zhimg.com/v2-226f7b5260ef465fe52959adf1f337ce_r.jpg)

#### Key ideas
- [SimTrack](simtrack.md) simplifies the tracking mechanism with improvements
	- Replaces bipartite matching with simple read-off. --> This may be improved with nearest neighbor look up
	- track life management: previous feature map is added to the current frame, after ego motion compensation. If the memory fades, then 
- Architecture: can be used as an add-on to any voxel or pillar based lidar object detector (such as [PointPillars](point_pillars.md) and [VoxelNet](voxel_net.md)).
	- Hybrid time centerness map
	- Motion updating: predicting the movement between two consecutive frames, and used to update centerness map during inference. 
	- Regression: same as [CenterPoint](centerpoint.md).

#### Technical details
- The memory mechanism is not trained end to end, but rather there is a discrepancy between training and inference. --> this may be a future direction for improvement
- point input (x, y, z, r, $\Delta_t$) where $\Delta_t$ is the relative timestamp to the current sweep.
- Ground truth handling in hybrid map (first appear)
	- Continuous track in t-1 and t: heat map in t-1 as positive
	- Dead track in t: negative
	- New track in t: heat map in t as positive
- Tracking by detection still remains the predominant method in tracking.
- AdamW + one cycle LR, following [CenterPoint](centerpoint.md).
- Evaluation: [nuScenes](nuscenes.md) uses 2 m in BEV, and [Waymo](waymo_dataset.md) uses 0.7 3D IoU to denote TP.
- Input pillar size is 0.2 m, same as [CenterPoint](centerpoint.md) for apple-to-apple comparison.
- Additional latency is only 1-2 ms (on Titan RTX though).

#### Notes
- Code will be available on [github](https://github.com/qcraftai/simtrack).
- ​Two questions sent to the 1st author: 
	- I think the better performance of SimTrack to handle occlusion does not only come from the better track management, but more from the operation of concatenating previous updated centerness map. This resembles a hand-crafted LSTM in inference time and introduces long term memory. If we remove the combined map (as discussed in 4.5), would SimTrack still be able to handle occlusions? My guess is that it would be no better than CenterPoint.
	- ​The simple operation of read-off (or look-up) is implicitly based on the assumption that the prediction is accurate enough to make the prediction (detection at t-1 plus motion) and the detection of the next frame (detection at t) fall under the same feature map grid. Essentially it also has an implicit threshold -- the grid size of the centerness feature map (instead of an explicit one as in CenterPoint, 1 m, 4 m, etc).
- SimTrack improves both detection and tracking. --> Honestly I think the detection improvement is the key to the success of this paper. 


