# Andrej Karpathy's Talk

## [ICML 2019 (CVPR 2019)](https://www.youtube.com/watch?v=IHH47nZ7FZU)

## [ScaledML 2020](https://www.youtube.com/watch?v=hx7BXih7zx8)

- [Review by 黄浴](https://zhuanlan.zhihu.com/p/136179627)

Stuff that caught my eye:

- Even state-less SOD such as stop signs can be complex
	- active states and modifiers
![](../assets/scaledml_2020/stop_overview.jpg)
- temporal flickering in shadow mode indicates corner case
- Test driven feature development
![](../assets/scaledml_2020/evaluation.jpg)
- BEVNet to learn local map from camera images
![](../assets/scaledml_2020/bevnet.jpg)
- Pseudo-lidar (Vidar) approach is promising in urban driving (40mx40m range)
![](../assets/scaledml_2020/vidar.jpg)
- infrastructure: operational vacation
![](../assets/scaledml_2020/operation_vacation.jpg)
- Other pics
![](../assets/scaledml_2020/stop1.jpg)
![](../assets/scaledml_2020/stop2.jpg)
![](../assets/scaledml_2020/stop3.jpg)
![](../assets/scaledml_2020/stop4.jpg)
![](../assets/scaledml_2020/stop5.jpg)
![](../assets/scaledml_2020/stop6.jpg)
![](../assets/scaledml_2020/stop7.jpg)
![](../assets/scaledml_2020/stop8.jpg)
![](../assets/scaledml_2020/stop9.jpg)
![](../assets/scaledml_2020/stop10.jpg)
![](../assets/scaledml_2020/stop11.jpg)
![](../assets/scaledml_2020/stop12.jpg)
![](../assets/scaledml_2020/stop13.jpg)
![](../assets/scaledml_2020/env.jpg)
![](../assets/scaledml_2020/pedestrian_aeb.jpg)


## [CVPR 2020](https://www.youtube.com/watch?v=g2R2T631x7k)
- [link in my Meeting Notes repo](https://github.com/patrick-llgc/MeetingNotes/blob/master/CVPR2020/workshops.md#scalability-in-autonomous-driving-video-on-youtube)


## [Pytorch Conf](https://www.youtube.com/watch?v=hx7BXih7zx8)
- [A very good review blog here](https://phucnsp.github.io/blog/self-taught/2020/04/30/tesla-nn-in-production.html)

## [2021.06.20 CVPR 2021](https://www.youtube.com/watch?v=g6bOwQdCJrc)
- The grand mission: Tesla is ditching radars. They are using neural network and vision to do radar depth + velocity sensing.
- In order to do that, they need a large AND diverse 4D (3D+time) dataset. This is also used to train FSD. 
- Tesla has a whole team spending about 4 months focusing on autolabeling 
- Tesla uses MANY (221 as of mid-2021) triggers to collect the diverse dataset. They ended up with 1 million 10-second clips.
- Dedicated HPC team. Now Tesla training with 720 8-GPU nodes!
- Tesla argues that vision alone is perfectly capable of depth sensing. It is hard and it requires the fleet.
![](../assets/cvpr_2021_andrej/cover.jpg)




PMM: pedal misuse mitigation
![](../assets/cvpr_2021_andrej/traffic_control_warning_pmm.jpg)

Tesla's data set-up.
![](../assets/cvpr_2021_andrej/tesla_no_radar.jpg)
![](../assets/cvpr_2021_andrej/8cam_setup.jpg)
![](../assets/cvpr_2021_andrej/large_clean_diverse_data.jpg)

Have to figure out the road layout the first time the car goes there (drive on perception). Fundamental problem: Depth estimation of monocular 
![](../assets/cvpr_2021_andrej/data_auto_labeling.jpg)
![](../assets/cvpr_2021_andrej/trainig_cluster.jpg)
![](../assets/cvpr_2021_andrej/tesla_dataset.jpg)

Once in a while radar gives you a FP that is hard to handle
![](../assets/cvpr_2021_andrej/depth_velocity_with_vision_1.jpg)
![](../assets/cvpr_2021_andrej/depth_velocity_with_vision_2.jpg)
![](../assets/cvpr_2021_andrej/depth_velocity_with_vision_3.jpg)

Validation process
![](../assets/cvpr_2021_andrej/release_and_validation.jpg)

## [Tesla Patents](https://patents.google.com/?q=(machine+learning)&assignee=Tesla%2c+Inc.&after=priority:20180101&oq=(machine+learning)+assignee:(Tesla%2c+Inc.)+after:priority:20180101)

## On FSD
- [Tweets from @phlhr](https://twitter.com/phlhr/status/1318335219586326529) and [another one](https://twitter.com/phlhr/status/1357924763214049285)