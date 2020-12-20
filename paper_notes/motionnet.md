# [MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps](https://arxiv.org/abs/2003.06754)

_December 2020_

tl;dr: Predict semantic class and motion of BEV occupancy grid, serving as a backup for 3D object detection.

#### Overall impression
Data-driven lidar 3D object detection cannot handle unseen corner cases, and occupancy grid map (OGM) does not have semantic information and does not have correspondence of cells across time and thus hard to reason object-level dynamics. ([Monocular BEV semantic segmentation](../topic_bev_segmentation.md) also has this drawbacks).

MotionNet proposes to use a BEV representation that extends OGM by occupancy, motion, and category information. Motion info is encoded by associating each cell with displacement vectors, as in [CenterPoint](centerpoint.md)

#### Key ideas
- Motion prediction without bbox
	- Each grid cell has cls, state (static/motion) and motion forecasting
	- More general than object as object as object detection relies on texture/shape info of instances in training set. This will not generalize well to unseen objects. 
	- Breaking down to each occupancy cell and focus on motion cues will lead to better generalization
	- Serving as a backup to the bounding box based systems. 
- Input/Output representation
	- binary occupancy grid, similar to that of [PIXOR](pixor.md).
	- Motion is represented as NxHxWx2, N steps. Similar to the [CenterNet](centernet.md) head design.
- **Spatial/temporal pyramid network (STPN)**
	- Spatiotemporal convolution (STC) operation
	- Input: TxCxHxW, C is height dim
	- Spatial pooling followed by temporal 1x1xT conv
	- Seems to be inspired by [TSM](tsm.md) but works better.
	- Better than LSTM as well as training is more efficient (no need to do multi-step backprop)
- Consistency loss
	- Class and motion status are trained with focal loss
	- Spatial consistency loss: grid of the same object should have similar motion
	- Foreground temporal consistency loss: objects should have smooth motion over time
	- Background temporal consistency loss: transformed background should have similar motion --> Isn't this motion always zero?
- Postprocessing by filtering out background grids and parked grids. 

#### Technical details
- Motion compensation is key. MC with IMU this is better than MC with ICP algorithm.
- Preprocessing of [nuScenes](nuscenes.md) annotation into MotionNet's motion GT: 
	- For points inside bbox: motion is $Rx + c\Delta - x$, R is the yaw rotation wrt bbox center, and $c\Delta$ is the displacement of box center.
	- For points outside bbox: 0
- Temporal aggregation: 4 unannotated + 1 annotated
- Predict relative displacement between neighboring frames.
- Multitask training with [MGDA (multiple gradient descent algorithm)](https://arxiv.org/abs/1810.04650) <kbd>NeurIPS 2018</kbd> works better than human crafted weights.
- Middle fusion as in STPN works better than early fusion or late fusion
	- Early fusion squashes temporal info too early and fails to capture fast moving objects
	- Late fusion algorithm ignores too many low-level motion cues.

#### Notes
- [MotionNet on Github](https://github.com/pxiangwu/MotionNet)
- Maybe combining MotionNet and Monocular BEV semantic segmentation is the future. 

