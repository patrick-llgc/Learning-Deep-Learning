# [PnPNet: End-to-End Perception and Prediction with Tracking in the Loop](https://arxiv.org/abs/2005.14711)

_September 2021_

tl;dr: Have tracker in the loop improves perception and prediction. Track level features is important for long term trajectory prediction.

#### Overall impression
MOT has two challenges: the discrete problem of data association and the continuous problem of trajectory estimation.

Previous methods with perception and prediction only **uses tracking as postprocessing.** The full temporal history contained in tracks is not used by detection and prediction. They usually limit the time step to 3, instead of a long-term trajectory. Their performance usually **saturates** with fewer than 1 second of sensor data.

![](https://lfs.aminer.cn/upload/pdf_image/5ecc/534/5eccb534e06a4c1b26a8391dimg-001.png)

PnPNet includes a tracker in the loop and thus can be trained end to end. The Hungarian matching cost function is learnable.

#### Key ideas
- Tracker in the loop to leverage long-term track level information.
- Input: multiple sweeps by concatenating along the height dimension, with the ego motion compensated for the previous sweeps, similar to [IntentNet](intentnet.md).
- Trajectory level object trajectory
	- Start from memories of BEV features maps
	- Based on the object location, feature maps are [rotated-RoI-pooled](https://youtu.be/IpfcK612swo?t=1637) from the BEV feature map
	- Velocity is obtained by finite difference.
	- An LSTM to mine features in the track $h(P^t_j)$
- Motion forecasting is based on the track feature $h(P^t_j)$. This is different from [FaF](faf.md) and [IntentNet](intentnet.md). **Exploting motion from explicit object trajectories is more accurate than inferring motion from the features computed from the raw sensor data.**

#### Technical details
- Prediction = Motion Forceasting.
- Track history is 16 frames @ 10 Hz, larger than 1 second.
- The RRoIAlign is also mentioned in [Spatially-Aware Graph Neural Networks for Relational Behavior Forecasting from Sensor Data](https://arxiv.org/abs/1910.08233) <kbd>ICRA 2020</kbd>. See [youtube video](https://youtu.be/IpfcK612swo?t=1289) for a good illustration.
![](https://cdn-images-1.medium.com/max/1600/1*sYM1ROauG3-3hu443tVCjg.png)

#### Notes
- [Raquel's talk's recap on tracking on 知乎](https://zhuanlan.zhihu.com/p/151094211)
- [1 min video on Youtube](https://www.youtube.com/watch?v=XIaNTrfcl5s)
- [Tracking for Self-Driving Cars](https://www.youtube.com/watch?v=IpfcK612swo)
