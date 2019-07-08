# [Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)

_July 2019_

tl;dr: A single network to do detection, tracking and prediction.

#### Overall impression
The [oral presentation](https://youtu.be/Jl1NeziAHFY?t=1471) is quite impressive. Modern approaches to autonomous driving has four steps: detection, tracking, motion forecasting and planning.

The assumption of the paper is that tracking and prediction can help object detection, reducing both false positives and false negatives.

More robust to occlusion and sparse data at range. It also runs real-time at 33 FPS.

#### Key ideas
- Two fusion strategies:
	- Early fusion: fuse time dimension from the beginning. This is essentially doing a temporal averaging of all frames. 
	- Later fusion: use 3x3x3 in two layers to reduce temporal dimension from 5 to 1. This strategy yields better results than early fusion. Note that no LSTM is used in the process. 
- **Decodes tracklets from prediction by average pooling.**
	- Each timestamp will have current detection and n-1 past predictions.
	- If detection and motion prediction are perfect, we can have perfect decoding of tracklets. When the past's prediction and current detection have overlaps, it is considered to be the same object and bboxes are averaged. 
- Adding temporal information by taking all 3D points from past n frames. Motion frames need to be ego-motion compensated. 

#### Technical details
- BEV representation is metric and thus prior knowledge about cars can be exploited. In total 6 anchors are used at 5 m and 8 m scale. (using anchor boxes reduce the variance of regression target thus making the network easy to train).
- 6 regression targets: location offset x, y, log-normalized sizes w, h and heading parameters sin and cos.
- Vehicles with more than 3 points in the 3D bbox are evaluated. Otherwise it is counted as **"don't care"**.
- Tacking KPIs such as MOTA, MOTP, MT (mostly tracked) and ML (mostly lost).

#### Notes
- How long in seconds can this predict in future? 10 frames does not mean anything.
- Check the KITTI KPI [evaluation criterion on don't care region](http://www.cvlibs.net/datasets/kitti/eval_object.php).