# [QCNet: Query-Centric Trajectory Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf)

_June 2024_

tl;dr: Query centic prediction that marries agent centric and scene centric predictions.

#### Overall impression
Winning solution in Argoverse and Waymo datasets. 

#### Key ideas
- Local coordinate system for each agent that leverages invariance.
- Long horizon prediction in 6-8s is achieved by AR decoding of 1s each, then followed by a trajectory refiner. --> This means the target oriented approach scuh as [TNT](tnt.md) might have been too hard. [TNT](tnt.md) seems to have been proposed to maximize FDE directly.

#### Technical details
- Summary of technical details, such as important training details, or bugs of previous benchmarks.

#### Notes
- [Tech blog in Chinese by 周梓康](https://mp.weixin.qq.com/s/Aek1ThqbrKWCSMHG6Xr9eA)

