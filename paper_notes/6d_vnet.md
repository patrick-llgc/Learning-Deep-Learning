# [6D-VNet: End-to-end 6DoF Vehicle Pose Estimation from Monocular RGB Images](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Autonomous%20Driving/Wu_6D-VNet_End-to-End_6-DoF_Vehicle_Pose_Estimation_From_Monocular_RGB_Images_CVPRW_2019_paper.pdf) 

_January 2020_

tl;dr: Directly regress 3D distance and quaternion direction from RoIPooled features. 

#### Overall impression
This is an extension of mask RCNN, by extending the mask head to regress fine-grained vehicle model (such as Audi Q5), quaternion and distance.

#### Key ideas
- Previous methods usually estimate depth via two step process: 1) regress bbox and direction 2) postprocess to estimate 3D translation via projective distance estimation. --> this requires bbox and orientation to be estimated correctly.
- Robotics usually requires strict estimation of orientation but translation can be relaxed. However AD require accurate estimation of translation.
- the features for regreessing fine-class and orientation (quaternion) is also concate with the translational branch to predict translation. 
- The target of translation is also preprocessed to essentially regress z directly. This can also be used to predict the 3D projection.
$$
x = (u - c_x) z / f_x \\
y = (u - c_y) z / f_y 
$$

#### Technical details
- Summary of technical details

#### Notes
- Postprocessing DL output may suffer from error accumulation. If we work the postprocessing into label preprocessing, this could not be a problem anymore. Of course, keeping both will add to redundancy.
- [github code](https://github.com/stevenwudi/6DVNET)

