# Paper notes
This repository contains my paper reading notes on deep learning and machine learning. It is inspired by [Denny Britz](https://github.com/dennybritz/deeplearning-papernotes) and [Daniel Takeshi](https://github.com/DanielTakeshi/Paper_Notes).

**New year resolution for 2019: read at least one paper a week!**

## Summary of Topics

| Topics                                      |
| ------------------------------------------- |
| <kbd>DRL</kbd>  Deep Reinforcement Learning |
| <kbd>CLS</kbd>  Classification              |
| <kbd>OD</kbd>  Object Detection             |
| <kbd>InsSeg</kbd>, <kbd>SemSeg</kbd>, <kbd>PanSeg</kbd>  Segmentation |
| <kbd>Video</kbd> Video understanding |
| <kbd>MI</kbd>  Medical Imaging |
| <kbd>NIPS</kbd>, <kbd>CVPR</kbd>, <kbd>ICCV</kbd>, <kbd>ECCV</kbd> Conferences |
| <kbd>Mono3DOD</kbd> Monocular 3D Object Detection |
| <kbd>MonoDepEst</kbd> Monocular Depth Estimation |

The sections below records paper reading activity in chronological order. See notes organized according to subfields [here](organized.md) (up to 06-2019). 


## 2019-08 (4)
- [Mono3D: Monocular 3D Object Detection for Autonomous Driving](https://www.cs.toronto.edu/~urtasun/publications/chen_etal_cvpr16.pdf) [[Notes](paper_notes/mono3d.md)] <kbd>CVPR2016</kbd>
- [MonoDIS: Disentangling Monocular 3D Object Detection](https://arxiv.org/abs/1905.12365) [[Notes](paper_notes/monodis.md)] <kbd>ICCV 2019</kbd>
- [Pseudo lidar-e2e: Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud](https://arxiv.org/abs/1903.09847) [[Notes](paper_notes/pseudo_lidar_e2e.md)] (pseudo-lidar with 2d and 3d consistency loss, better than PL and worse than PL++, SOTA for pure mono3D)
- [MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization](https://arxiv.org/pdf/1811.10247.pdf) [[Notes](paper_notes/monogrnet.md)] <kbd>AAAI 2019</kbd> (SOTA of Mono3DOD, MLF < MonoGRNet < Pseudo-lidar)
- [MLF: Multi-Level Fusion based 3D Object Detection from Monocular Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Multi-Level_Fusion_Based_CVPR_2018_paper.pdf) [[Notes](paper_notes/mlf.md)] <kbd>CVPR 2018</kbd> (precursor to pseudo-lidar)
- [ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape](https://arxiv.org/abs/1812.02781) [[Notes](paper_notes/roi10d.md)] <kbd>CVPR 2019</kbd>
- [RoarNet: A Robust 3D Object Detection based on RegiOn Approximation Refinement](https://arxiv.org/abs/1811.03818) (3D mono proposal, refined in point cloud)
- [Mono3D++: Monocular 3D Vehicle Detection with Two-Scale 3D Hypotheses and Task Priors](https://arxiv.org/abs/1901.03446) (from Stefano Soatto) <kbd>AAAI 2018</kbd>
- [3DOP: 3D Object Proposals for Accurate Object Class Detection](https://papers.nips.cc/paper/5644-3d-object-proposals-for-accurate-object-class-detection) <kbd>NIPS 2015</kbd>
- [Accurate Monocular 3D Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving](https://arxiv.org/pdf/1903.11444.pdf) (similar to pseudo-lidar)
- [3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kundu_3D-RCNN_Instance-Level_3D_CVPR_2018_paper.pdf) (from Uber ATG)
- [Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)
- [Detect to Track and Track to Detect](https://arxiv.org/abs/1710.03958) <kbd>ICCV 2017</kbd> (from Christoph Feichtenhofer)
- [A Novel Approach for Detecting Road Based on Two-Stream Fusion Fully Convolutional Network]() (convert camera to BEV)
- [Classification of Objects in Polarimetric Radar Images Using CNNs at 77 GHz](http://sci-hub.tw/10.1109/APMC.2017.8251453) (Radar, polar)
- [Distant Vehicle Detection Using Radar and Vision](https://arxiv.org/abs/1901.10951) [[Notes]()] <-- todo!
- [PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation](https://arxiv.org/pdf/1711.10871.pdf) <kbd>CVPR 2018</kbd>
- [A Survey on 3D Object Detection Methods for Autonomous Driving Applications](http://wrap.warwick.ac.uk/114314/1/WRAP-survey-3D-object-detection-methods-autonomous-driving-applications-Arnold-2019.pdf) (Survey) <kbd>TITS 2019</kbd>
- [Eliminating the Blind Spot: Adapting 3D Object Detection and Monocular Depth Estimation to 360° Panoramic Imagery](https://arxiv.org/abs/1808.06253) <kbd>ECCV 2018</kbd> (Monocular 3D object detection and depth estimation)
- [DORN: Deep Ordinal Regression Network for Monocular Depth Estimation](https://arxiv.org/pdf/1806.02446.pdf) <kbd>CVPR 2018</kbd> 
- [In-Place Activated BatchNorm for Memory-Optimized Training of DNNs](https://arxiv.org/abs/1712.02616) <kbd>CVPR 2018</kbd> (optimized BatchNorm + ReLU)
- [The DriveU Traffic Light Dataset: Introduction and Comparison with Existing Datasets](https://ieeexplore.ieee.org/document/8460737) <kbd>ICRA 2018</kbd> 
- [Deep Metadata Fusion for Traffic Light to Lane Assignment](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8613841) <kbd>IEEE RA-L 2019</kbd> 
- [Shift R-CNN: Deep Monocular 3D Object Detection with Closed-Form Geometric Constraints](https://arxiv.org/abs/1905.09970) <kbd>IEEE ICIP</kbd>
- [Triangulation Learning Network: from Monocular to Stereo 3D Object Detection](https://arxiv.org/abs/1906.01193) <kbd>CVPR 2019</kbd>
- [Deep Optics for Monocular Depth Estimation and 3D Object Detection](https://arxiv.org/abs/1904.08601)
- [Monocular 3D Object Detection via Geometric Reasoning on Keypoints](https://arxiv.org/abs/1905.05618)
- [Deep Fitting Degree Scoring Network for Monocular 3D Object Detection](https://arxiv.org/abs/1904.12681) <kbd>CVPR 2019</kbd>
- [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](https://arxiv.org/abs/1903.10955) <kbd>CVPR 2019</kbd>
- [Monocular 3D Object Detection and Box Fitting Trained End-to-End Using Intersection-over-Union Loss](https://arxiv.org/abs/1906.08070)
- [DirectShape: Photometric Alignment of Shape Priors for Visual Vehicle Pose and Shape Estimation](https://arxiv.org/abs/1904.10097)
- [M3D-RPN: Monocular 3D Region Proposal Network for Object Detection](https://arxiv.org/abs/1907.06038) (Xiaoming Liu)
- [Learning 2D to 3D Lifting for Object Detection in 3D for Autonomous Vehicles](https://arxiv.org/abs/1904.08494) (BirdGAN)
- [an intrigging faiing of convolutional neural networks and the cordconv solution]()

## 2019-07 (19)
- [Deep Parametric Continuous Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf) [[Notes](paper_notes/parametric_cont_conv.md)] <kbd>CVPR 2018</kbd> (@Uber, sensor fusion)
- [ContFuse: Deep Continuous Fusion for Multi-Sensor 3D Object Detection](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf) [[Notes](paper_notes/contfuse.md)] <kbd>ECCV 2018</kbd> (@Uber, sensor fusion, birds eye view)
- [Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf) [[Notes](paper_notes/faf.md)] <kbd>CVPR 2018 oral</kbd> (lidar only, perception and prediction)
- [Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/pdf/1904.04998.pdf) \[[Notes](paper_notes/mono_depth_video_in_the_wild.md)] (monocular depth estimation, intrinsic estimation, SOTA)
- [monodepth: Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://arxiv.org/abs/1609.03677) [[Notes](paper_notes/monodepth.md)] <kbd>CVPR 2017 oral</kbd> (monocular depth estimation, stereo for training)
- [Struct2depth: Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/pdf/1811.06152.pdf) [[Notes](paper_notes/struct2depth.md)] <kbd>AAAI 2019</kbd> (monocular depth estimation, estimating movement of dynamic object)
- [Unsupervised Learning of Geometry with Edge-aware Depth-Normal Consistency](https://arxiv.org/pdf/1711.03665.pdf) [[Notes](paper_notes/edge_aware_depth_normal.md)] <kbd>AAAI 2018</kbd> (monocular depth estimation, static assumption, surface normal)
- [LEGO Learning Edge with Geometry all at Once by Watching Videos](https://arxiv.org/pdf/1803.05648.pdf) [[Notes](paper_notes/lego.md)] <kbd>CVPR 2018 Spotlight</kbd> (monocular depth estimation, static assumption, surface normal)
- [Object Detection and 3D Estimation via an FMCW Radar Using a Fully Convolutional Network](https://arxiv.org/abs/1902.05394) [[Notes](paper_notes/radar_3d_od_fcn.md)] (radar, RD map, OD, Arxiv 201902) 
- [A study on Radar Target Detection Based on Deep Neural Networks](https://www.researchgate.net/publication/330748053_A_Study_on_Radar_Target_Detection_Based_on_Deep_Neural_Networks) [[Notes](paper_notes/radar_target_detection_tsinghua.md)] (radar, RD map, OD) 
- [2D Car Detection in Radar Data with PointNets](https://arxiv.org/abs/1904.08414) [[Notes](paper_notes/radar_detection_pointnet.md)] (from Ulm Univ, radar, point cloud, OD, Arxiv 201904) 
- [Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/abs/1802.04865) [[Notes](paper_notes/learning_ood_conf.md)] (budget to cheat)
- [A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification](assets/papers/bosch_traffic_lights.pdf) [[Notes](paper_notes/bosch_traffic_lights.md)] <kbd>ICRA 2017</kbd> (Bosch, traffic lights)
- [How hard can it be? Estimating the difficulty of visual search in an image](https://arxiv.org/abs/1705.08280) [[Notes](paper_notes/how_hard_can_it_be.md)] <kbd>CVPR 2016</kbd>
- [Deep Multi-modal Object Detection and Semantic Segmentation for Autonomous Driving: Datasets, Methods, and Challenges](https://arxiv.org/pdf/1902.07830.pdf) [[Notes](paper_notes/deep_fusion_review.md)] (review from Bosch)
- [Review of monocular 3d object detection](https://zhuanlan.zhihu.com/p/57029694) (blog from 知乎)
- [Deep3dBox: 3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/pdf/1612.00496.pdf) [[Notes](paper_notes/deep3dbox.md)] (from Zoox) <kbd>CVPR 2017</kbd>
- [MonoPSR: Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction](https://arxiv.org/abs/1904.01690) [[Notes](paper_notes/monopsr.md)] <kbd>CVPR 2019</kbd>
- [OFT: Orthographic Feature Transform for Monocular 3D Object Detection](https://arxiv.org/pdf/1811.08188.pdf) [[Notes](paper_notes/oft.md)] (Convert camera to BEV, Alex Kendall) <kbd>BMVC 2019</kbd>


## 2019-06 (12)
- [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249) [[Notes](paper_notes/MixMatch.md)]
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) [[Notes](paper_notes/efficientnet.md)] <kbd>ICML 2019</kbd>
- [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/pdf/1703.04977.pdf) [[Notes](paper_notes/uncertainty_bdl.md)] <kbd>NIPS 2017</kbd>
- [Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding](https://arxiv.org/pdf/1511.02680.pdf) [[Notes](paper_notes/bayesian_segnet.md)]<kbd>BMVC 2017</kbd>
- [TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents](https://arxiv.org/pdf/1811.02146.pdf) [[Notes](paper_notes/trafficpredict.md)] <kbd>AAAI 2019 (oral)</kbd>
- [Deep Depth Completion of a Single RGB-D Image](https://arxiv.org/pdf/1803.09326.pdf) [[Notes](paper_notes/deep_depth_completion_rgbd.md)] <kbd>CVPR 2018</kbd> (indoor)
- [DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image](https://arxiv.org/pdf/1812.00488v2.pdf) [[Notes](paper_notes/deeplidar.md)] <kbd>CVPR 2019</kbd> (outdoor)
- [SfMLearner: Unsupervised Learning of Depth and Ego-Motion from Video](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf) [[Notes](paper_notes/sfm_learner.md)] <kbd>CVPR 2017</kbd>
- [Monodepth2: Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260) [[Notes](paper_notes/monodepth2.md)] \(@Niantic)
- [DeepSignals: Predicting Intent of Drivers Through Visual Signals](https://arxiv.org/pdf/1905.01333.pdf) [[Notes](paper_notes/deep_signals.md)] <kbd>ICRA2019</kbd> (@Uber, turn signal detection)
- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf) [[Notes](paper_notes/fcos.md)]
- [Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving](https://arxiv.org/pdf/1906.06310.pdf) [[Notes](paper_notes/pseudo_lidar++.md)]
- [MMF: Multi-Task Multi-Sensor Fusion for 3D Object Detection](http://www.cs.toronto.edu/~byang/papers/mmf.pdf) [[Notes](paper_notes/mmf.md)] <kbd>CVPR 2019</kbd> (@Uber, sensor fusion)


## 2019-05 (18)
- [CenterNet: Objects as points](https://arxiv.org/pdf/1904.07850.pdf) (from ExtremeNet authors) [[Notes](paper_notes/centernet_ut.md)]
- [CenterNet: Object Detection with Keypoint Triplets](https://arxiv.org/pdf/1904.08189.pdf) [[Notes](paper_notes/centernet_cas.md)]
- [Object Detection based on Region Decomposition and Assembly](https://arxiv.org/pdf/1901.08225.pdf) [[Notes](paper_notes/object_detection_region_decomposition.md)] <kbd> AAAI 2019 </kbd>
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) [[Notes](paper_notes/lottery_ticket_hypothesis.md)] <kbd> ICLR 2019 </kbd>
- [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/abs/1811.04533) [[Notes](paper_notes/m2det.md)] <kbd> AAAI 2019 </kbd>
- [Deep Radar Detector](assets/papers/deep_radar_detector.pdf) [[Notes](paper_notes/deep_radar_detector.md)] <kbd> RadarCon 2019</kbd>
- [Semantic Segmentation on Radar Point Clouds](https://ieeexplore.ieee.org/document/8455344) [[[Notes](paper_notes/radar_point_semantic_seg.md)]] (from Daimler AG) <kbd> FUSION 2018</kbd>
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/pdf/1608.08710.pdf) [[Notes](paper_notes/pruning_filters.md)] <kbd>ICLR 2017</kbd>
- [Layer-compensated Pruning for Resource-constrained Convolutional Neural Networks](https://arxiv.org/pdf/1810.00518.pdf) [[Notes](paper_notes/layer_compensated_pruning.md)] <kbd>NIPS 2018 Talk</kbd>
- [LeGR: Filter Pruning via Learned Global Ranking](https://arxiv.org/pdf/1904.12368.pdf) [[Notes](paper_notes/legr.md)]
- [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](https://arxiv.org/pdf/1904.07392.pdf) [[Notes](paper_notes/nas_fpn.md)] <kbd> CVPR 2019 </kbd>
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501) [[Notes](paper_notes/autoaugment.md)] <kbd> CVPR 2019 </kbd>
- [Path Aggregation Network for Instance Segmentation](https://arxiv.org/pdf/1803.01534.pdf) [[Notes](paper_notes/panet.md)] <kbd> CVPR 2018 </kbd>
- [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/pdf/1707.06168.pdf) <kbd>ICCV 2017</kbd> (Face++, Yihui He) [[Notes](paper_notes/channel_pruning_megvii.md)]
- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf) <kbd>ECCV 2018</kbd> (Song Han, Yihui He)
- [MobileNetV3: Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) [[Notes](paper_notes/mobilenets_v3.md)]
- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf) [[Notes](mnasnet.md)] <kbd>CVPR 2019</kbd> 
- [Rethinking the Value of Network Pruning](https://arxiv.org/pdf/1810.05270.pdf) <kbd>ICLR 2019</kbd>

## 2019-04 (12)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf) (MobileNets v2) [[Notes](paper_notes/mobilenets_v2.md)] <kbd>CVPR 2018</kbd>
- [A New Performance Measure and Evaluation Benchmark
for Road Detection Algorithms](http://www.cvlibs.net/publications/Fritsch2013ITSC.pdf) [[Notes](paper_notes/kitti_lane.md)] <kbd>ITSC 2013</kbd>
- [MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving](https://arxiv.org/pdf/1612.07695.pdf) [[Notes](paper_notes/multinet_raquel.md)]
- [Optimizing the Trade-off between Single-Stage and Two-Stage Object Detectors using Image Difficulty Prediction](https://arxiv.org/pdf/1803.08707.pdf) (Very nice illustration of 1 and 2 stage object detection)
- [Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://arxiv.org/pdf/1711.07264.pdf) (Megvii) [[Notes](paper_notes/lighthead_rcnn.md)]
- [CSP: High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection](https://arxiv.org/pdf/1904.02948.pdf) (center and scale prediction) [[Notes](paper_notes/csp_pedestrian.md)] <kbd>CVPR 2019</kbd> 
- Review of Anchor-free methods (知乎Blog) [目标检测：Anchor-Free时代](https://zhuanlan.zhihu.com/p/62103812) [Anchor free深度学习的目标检测方法](https://zhuanlan.zhihu.com/p/64563186) [My Slides on CSP](https://docs.google.com/presentation/d/1_dUfxv63108bZXUnVYPIOAdEIkRZw5BR9-rOp-Ni0X0/)
- [DenseBox: Unifying Landmark Localization with End to End Object Detection](https://arxiv.org/pdf/1509.04874.pdf)
- [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/pdf/1808.01244.pdf) [[Notes](paper_notes/cornernet.md)] <kbd>ECCV 2018</kbd>
- [ExtremeNet: Bottom-up Object Detection by Grouping Extreme and Center Points](https://arxiv.org/pdf/1901.08043.pdf) [[Notes](paper_notes/extremenet.md)] <kbd>CVPR 2019</kbd>
- [FSAF: Feature Selective Anchor-Free Module for Single-Shot Object Detection](https://arxiv.org/pdf/1903.00621.pdf) [[Notes](paper_notes/fsaf_detection.md)] <kbd>CVPR 2019</kbd>
- [FoveaBox: Beyond Anchor-based Object Detector](https://arxiv.org/pdf/1904.03797v1.pdf) (anchor-free) [[Notes](paper_notes/foveabox.md)]



## 2019-03 (19)
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf) [[Notes](paper_notes/bag_of_freebies_object_detection.md)]
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf) [[Notes](paper_notes/mixup.md)] <kbd>ICLR 2018</kbd>
- [Multi-view Convolutional Neural Networks for 3D Shape Recognition](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf) (MVCNN) [[Notes](paper_notes/mvcnn.md)] <kbd>ICCV 2015</kbd> 
- [3D ShapeNets: A Deep Representation for Volumetric Shapes](http://3dshapenets.cs.princeton.edu/paper.pdf) [[Notes](paper_notes/3d_shapenets.md)] <kbd>CVPR 2015</kbd>
- [Volumetric and Multi-View CNNs for Object Classification on 3D Data](https://arxiv.org/pdf/1604.03265.pdf) [[Notes](paper_notes/vol_vs_mvcnn.md)] <kbd>CVPR 2016</kbd>
- [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) [[Notes](paper_notes/groupnorm.md)] <kbd>ECCV 2018</kbd>
- [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf) [[Notes](paper_notes/stn.md)] <kbd>NIPS 2015</kbd>
- [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf) (F-PointNet) [[Notes](paper_notes/frustum_pointnet.md)] <kbd>CVPR 2018</kbd> 
- [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/pdf/1801.07829.pdf) [[Notes](paper_notes/edgeconv.md)]
- [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/pdf/1812.04244v1.pdf) (SOTA for 3D object detection) [[Notes](paper_notes/point_rcnn.md)] <kbd>CVPR 2019</kbd>
- [Multi-View 3D Object Detection Network for Autonomous Driving](https://arxiv.org/pdf/1611.07759.pdf) (MV3D) [[Notes](paper_notes/mv3d.md)] <kbd>CVPR 2017</kbd> (Baidu, sensor fusion, BV proposal)
- [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/pdf/1712.02294.pdf) (AVOD) [[Notes](paper_notes/avod.md)] <kbd>IROS 2018</kbd> (sensor fusion, multiview proposal)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) [[Notes](paper_notes/mobilenets.md)]
- [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/1812.07179.pdf) [[Notes](paper_notes/pseudo_lidar.md)] <kbd>CVPR 2019</kbd>
- [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/pdf/1711.06396.pdf) <kbd>CVPR 2018</kbd> (Apple, first end-to-end point cloud encoding to grid)
- [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337/pdf) <kbd>Sensors 2018</kbd> (builds on VoxelNet)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/pdf/1812.05784.pdf) [[Notes](paper_notes/point_pillars.md)] <kbd>CVPR 2019</kbd> (builds on SECOND)
- [Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf) [[Notes](paper_notes/kitti.md)] <kbd>CVPR 2012</kbd>
- [Vision meets Robotics: The KITTI Dataset](http://ww.cvlibs.net/publications/Geiger2013IJRR.pdf) [[Notes](paper_notes/kitti.md)] <kbd>IJRR 2013</kbd>


## 2019-02 (9)
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf) (I3D) [[Notes](paper_notes/quo_vadis_i3d.md)]<kbd>Video</kbd> <kbd>CVPR 2017</kbd>
- [Initialization Strategies of Spatio-Temporal Convolutional Neural Networks](https://arxiv.org/pdf/1503.07274.pdf) [[Notes](paper_notes/quo_vadis_i3d.md)] <kbd>Video</kbd>
- [Detect-and-Track: Efficient Pose Estimation in Videos](https://arxiv.org/pdf/1712.09184.pdf) [[Notes](paper_notes/quo_vadis_i3d.md)] <kbd>ICCV 2017</kbd> <kbd>Video</kbd>
- [Deep Learning Based Rib Centerline Extraction and Labeling](https://arxiv.org/pdf/1809.07082) [[Notes](paper_notes/rib_centerline_philips.md)] <kbd>MI</kbd> <kbd>MICCAI 2018</kbd>
- [SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf) [[Notes](paper_notes/slowfast.md)] <kbd>Video</kbd>
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) (ResNeXt) [[Notes](paper_notes/resnext.md)] <kbd>CVPR 2017</kbd>
- [Beyond the pixel plane: sensing and learning in 3D](https://thegradient.pub/beyond-the-pixel-plane-sensing-and-learning-in-3d/) (blog, [中文版本](https://zhuanlan.zhihu.com/p/44386618))
- [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf) (VoxNet) [[Notes](paper_notes/voxnet.md)]
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf) <kbd>CVPR 2017</kbd> [[Notes](paper_notes/pointnet.md)]
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf) <kbd>NIPS 2017</kbd> [[Notes](paper_notes/pointnet++.md)]
- [Review of Geometric deep learning 几何深度学习前沿 (from 知乎)](https://zhuanlan.zhihu.com/p/36888114) (Up to CVPR 2018)


## 2019-01 (10)
- [Human-level control through deep reinforcement learning (Nature DQN paper)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) [[Notes](paper_notes/nature_dqn_paper.md)] <kbd>DRL</kbd>
- [Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection](https://arxiv.org/pdf/1811.08661.pdf) [[Notes](paper_notes/retina_unet.md)] <kbd>MI</kbd>
- [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf) [[Notes](paper_notes/panoptic_segmentation.md)] <kbd>PanSeg</kbd>
- [Panoptic Feature Pyramid Networks](https://arxiv.org/pdf/1901.02446.pdf) [[Notes](paper_notes/panoptic_fpn.md)] <kbd>PanSeg</kbd> 
- [Attention-guided Unified Network for Panoptic Segmentation](https://arxiv.org/pdf/1812.03904.pdf) [[Notes](paper_notes/AUNet_panoptic.md)] <kbd>PanSeg</kbd>
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf) [[Notes](paper_notes/bag_of_tricks_cnn.md)] <kbd>CLS</kbd>
- [Deep Reinforcement Learning for Vessel Centerline Tracing in Multi-modality 3D Volumes](https://link.springer.com/chapter/10.1007/978-3-030-00937-3_86) [[Notes](paper_notes/drl_vessel_centerline.md)] <kbd>DRL</kbd> <kbd>MI</kbd>
- [Deep Reinforcement Learning for Flappy Bird](http://cs229.stanford.edu/proj2015/362_report.pdf) [[Notes](paper_notes/drl_flappy.md)] <kbd>DRL</kbd>
- [Long-Term Feature Banks for Detailed Video Understanding](https://arxiv.org/pdf/1812.05038.pdf) [[Notes](paper_notes/long_term_feat_bank.md)] <kbd>Video</kbd> 
- [Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf) [[Notes](paper_notes/non_local_net.md)] <kbd>Video</kbd> <kbd>CVPR 2018</kbd>


## 2018
- [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/pdf/1712.00726.pdf)
- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf) (RetinaNet) [[Notes](paper_notes/focal_loss.md)]
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507) (SENet)
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
- [Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211.pdf)
- [Learning Region Features for Object Detection](https://arxiv.org/pdf/1803.07066.pdf)

## 2017 and before
- [Learning notes on Deep Learning](Learning_notes.md)
- [List of Papers on Machine Learning](List_of_Machine_Learning_Papers.md)
- [Notes of Literature Review on CNN in CV](paper_notes/cnn_papers.md) This is the notes for all the papers in the recommended list [here](papers_and_books_to_start.md)
- [Notes of Literature Review (Others)](misc.md)
- [Notes on how to set up DL/ML environment](ML_DL_environment_Setup.md)
- [Useful setup notes](installation_log.md)

## Papers to Read
Here is the list of papers waiting to be read. 
### Deep Learning in general
- [SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving](https://arxiv.org/pdf/1612.01051.pdf)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf)
- [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://openreview.net/forum?id=Bygh9j09KX) <kbd>ICML 2019</kbd>
- [Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet](https://openreview.net/forum?id=SkfMWhAqYQ) (BagNet) [blog](https://blog.evjang.com/2019/02/bagnet.html) <kbd>ICML 2019</kbd>
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820v2.pdf)
- [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf)


### 2D Object detection
- [Mask Scoring R-CNN](https://arxiv.org/pdf/1903.00241.pdf) <kbd>CVPR 2019</kbd>
- [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/pdf/1604.03540.pdf)

### Instance and Panoptic Segmentation
- [TensorMask: A Foundation for Dense Object Segmentation](https://arxiv.org/pdf/1903.12174.pdf)


### Video Understanding
- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767.pdf) (C3D)  <kbd> Video </kbd><kbd> ICCV 2015 </kbd>
- [AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions](https://arxiv.org/pdf/1705.08421.pdf)
- [Spatiotemporal Residual Networks for Video Action Recognition](https://arxiv.org/pdf/1611.02155.pdf) (decouple spatiotemporal) <kbd>NIPS 2016</kbd>
- [Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks](https://arxiv.org/pdf/1711.10305.pdf) (P3D, decouple spatiotemporal) <kbd>ICCV 2017</kbd>
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/pdf/1711.11248.pdf) (decouple spatiotemporal) <kbd>CVPR 2018</kbd>
- [Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://arxiv.org/pdf/1712.04851.pdf) (decouple spatiotemporal) <kbd>ECCV 2018</kbd>
- [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/pdf/1711.09577.pdf) <kbd>CVPR 2018</kbd>

### Orthoganal architecture improvements
- [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/pdf/1803.02579.pdf)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf)


### Reinforcement Learning
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) <kbd> NIPS 2013 </kbd>
- [Multi-Scale Deep Reinforcement Learning for Real-Time 3D-Landmark Detection in CT Scan](http://comaniciu.net/Papers/MultiscaleDeepReinforcementLearning_PAMI18.pdf)
- [An Artificial Agent for Robust Image Registration](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14751/14296)

### 3D Perception
- [3D-CNN：3D Convolutional Neural Networks for Landing Zone Detection from LiDAR](https://www.ri.cmu.edu/pub_files/2015/3/maturana-root.pdf)
- [Generative and Discriminative Voxel Modeling with Convolutional Neural Networks](https://arxiv.org/pdf/1608.04236.pdf)
- [Orientation-boosted Voxel Nets for 3D Object Recognition](https://arxiv.org/pdf/1604.03351.pdf) (ORION) <BMVC 2017>
- [GIFT: A Real-time and Scalable 3D Shape Search Engine](https://arxiv.org/pdf/1604.01879.pdf) <kbd>CVPR 2016</kbd>
- [3D Shape Segmentation with Projective Convolutional Networks](https://people.cs.umass.edu/~kalo/papers/shapepfcn/) (ShapePFCN)<kbd>CVPR 2017</kbd>
- [Learning Local Shape Descriptors from Part Correspondences With Multi-view Convolutional Networks](https://arxiv.org/pdf/1706.04496.pdf)
- [Open3D: A Modern Library for 3D Data Processing](http://www.open3d.org/wordpress/wp-content/paper.pdf)
- [Multimodal Deep Learning for Robust RGB-D Object Recognition](https://arxiv.org/pdf/1507.06821.pdf) <kbd>IROS 2015</kbd>
- [FlowNet3D: Learning Scene Flow in 3D Point Clouds](https://arxiv.org/pdf/1806.01411.pdf) <kbd>CVPR 2019</kbd>
- [Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling](https://arxiv.org/pdf/1712.06760.pdf) <kbd>CVPR 2018</kbd> (Neighbors Do Help: Deeply Exploiting Local Structures of Point Clouds)
- [PU-Net: Point Cloud Upsampling Network](https://arxiv.org/pdf/1801.06761.pdf) <kbd>CVPR 2018</kbd>
- [Recurrent Slice Networks for 3D Segmentation of Point Clouds](https://arxiv.org/pdf/1802.04402.pdf) <kbd>CVPR 2018</kbd>
- [SPLATNet: Sparse Lattice Networks for Point Cloud Processing](https://arxiv.org/pdf/1802.08275.pdf) <kbd>CVPR 2018</kbd>
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf) <kbd>NIPS 2016</kbd>
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) <kbd>ICLR 2017</kbd>
- [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks](https://arxiv.org/pdf/1704.06803.pdf) <kbd>NIPS 2017</kbd>
- [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) <kbd>ICLR 2018</kbd>
- [3D-SSD: Learning Hierarchical Features from RGB-D Images for Amodal 3D Object Detection](https://arxiv.org/pdf/1711.00238.pdf) (3D SSD)
- [Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models](https://arxiv.org/pdf/1704.01222.pdf) <kbd>ICCV 2017</kbd>
- [Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis](https://arxiv.org/pdf/1612.00101.pdf) <kbd>CVPR 2017</kbd>
- [IPOD: Intensive Point-based Object Detector for Point Cloud](https://arxiv.org/pdf/1812.05276.pdf)
- [Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images](https://cis.temple.edu/~latecki/Papers/DengCVPR2017.pdf) <kbd>CVPR 2017</kbd>
- [2D-Driven 3D Object Detection in RGB-D Images](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lahoud_2D-Driven_3D_Object_ICCV_2017_paper.pdf)
- [3D-SSD: Learning Hierarchical Features from RGB-D Images for Amodal 3D Object Detection](https://arxiv.org/pdf/1711.00238.pdf)


### NLP
- [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/pdf/1404.2188.pdf) <kbd>ACL 2014</kbd>
- [FastText: Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf) <kbd>ACL 2017</kbd>
- [Siamese recurrent architectures for learning sentence similarity](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023) <kbd>AAAI 2016</kbd>
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) <kbd>ICLR 2013</kbd>
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) <kbd>ICLR 2015</kbd>
- [Transformers: Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) <kbd>NIPS 2017</kbd>

### Unsorted
- [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/abs/1904.08889) (from the authors of PointNet)
- [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/pdf/1801.07791.pdf) <kbd>NIPS 2018</kbd>
- [Learning Depth with Convolutional Spatial Propagation Network](https://arxiv.org/abs/1810.02695) (Baidu, depth from SPN) <kbd>ECCV 2018</kbd>
- [GNN tutorial at CVPR 2019](https://xiaolonw.github.io/graphnn/)
- [Model Vulnerability to Distributional Shifts over Image Transformation Sets](https://arxiv.org/pdf/1903.11900.pdf) (CVPR workshop) [tl:dr](https://www.reddit.com/r/MachineLearning/comments/b81uwq/r_model_vulnerability_to_distributional_shifts/)
- [A Unified Panoptic Segmentation Network](https://arxiv.org/pdf/1901.03784.pdf) <kbd>CVPR 2019</kbd> <kbd>PanSeg</kbd>
- [FastDraw: Addressing the Long Tail of Lane Detection by Adapting a Sequential Prediction Network](https://arxiv.org/abs/1905.04354)
- [PSMNet: Pyramid Stereo Matching Network](https://arxiv.org/pdf/1803.08669.pdf) <kbd>CVPR 2018</kbd>
- [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/pdf/1812.03079.pdf) (Waymo)
- [Stereo R-CNN based 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/1902.09738.pdf) <kbd>CVPR 2019</kbd>
- [Deep Rigid Instance Scene Flow](https://people.csail.mit.edu/weichium/papers/cvpr19-dsisf/paper.pdf) <kbd>CVPR 2019</kbd>
- [GeoNet: Deep Geodesic Networks for Point Cloud Analysis](https://arxiv.org/pdf/1901.00680.pdf) <kbd>CVPR 2019</kbd> (oral, Megvii)
- [StixelNet: A Deep Convolutional Network for Obstacle Detection and Road Segmentation](http://www.bmva.org/bmvc/2015/papers/paper109/paper109.pdf)
- [DenseBox: Unifying Landmark Localization with End to End Object Detection](https://arxiv.org/pdf/1509.04874.pdf)
- [Calibration of Heterogeneous Sensor Systems](https://arxiv.org/pdf/1812.11445.pdf)
- [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [The ApolloScape Open Dataset for Autonomous Driving and its Application](https://arxiv.org/pdf/1803.06184.pdf) (dataset, point cloud)
- [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/pdf/1903.11027.pdf) (dataset, point cloud, radar)
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) (Xception)
- [DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map](https://arxiv.org/pdf/1805.04949.pdf) <kbd>CVPR 2018</kbd>
- [2D-Driven 3D Object Detection in RGB-D Images](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lahoud_2D-Driven_3D_Object_ICCV_2017_paper.pdf) <kbd>ICCV 2017</kbd>
- [A Multi-Sensor Fusion System for Moving Object Detection and Tracking in Urban Driving Environments](http://www.cs.cmu.edu/~youngwoo/doc/icra-14-sensor-fusion.pdf) <kbd>ICRA 2014</kbd>
- [PIXOR: Real-time 3D Object Detection from Point Clouds](https://arxiv.org/pdf/1902.06326.pdf) <kbd>CVPR 2018</kbd> (birds eye view)
- [PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation](https://arxiv.org/pdf/1807.00652.pdf) (pointnet alternative, backbone)
- [Vehicle Detection from 3D Lidar Using Fully Convolutional Network](https://arxiv.org/pdf/1608.07916.pdf) (VeloFCN) <kbd>RSS 2016</kbd> 
- [Intro：Sensor Fusion for Adas 无人驾驶中的数据融合 (from 知乎)](https://zhuanlan.zhihu.com/p/40967227) (Up to CVPR 2018)
- [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/pdf/1904.09664.pdf) (from Charles Qi)
- [Efficient Deep Learning Inference based on Model Compression](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w33/Zhang_Efficient_Deep_Learning_CVPR_2018_paper.pdf) (Model Compression)
- [FCNN: Fourier Convolutional Neural Networks](http://ecmlpkdd2017.ijs.si/papers/paperID11.pdf) (FFT as CNN)
- [Network pruning tutorial](https://jacobgil.github.io/deeplearning/pruning-deep-learning) (blog)
- [Visualizing the Loss Landscape of Neural Nets](https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf) <kbd>NIPS 2018</kbd>
- [A Survey on Neural Architecture Search](https://arxiv.org/pdf/1905.01392.pdf)
- [Automatic adaptation of object detectors to new domains using self-training](https://arxiv.org/pdf/1904.07305.pdf) <kbd>CVPR 2019</kbd> (find corner case and boost)
- [Missing Labels in Object Detection](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Weakly%20Supervised%20Learning%20for%20Real-World%20Computer%20Vision%20Applications/Xu_Missing_Labels_in_Object_Detection_CVPRW_2019_paper.pdf) <kbd>CVPR 2019</kbd>
- [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (uncertainty)
- [Learning to Drive from Simulation without Real World Labels](https://arxiv.org/abs/1812.03823) <kbd>ICRA 2019</kbd> (domain adaptation, sim2real)
- [Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/pdf/1802.05522.pdf) <kbd>CVPR 2018</kbd>
- [YUVMultiNet: Real-time YUV multi-task CNN for autonomous driving](https://arxiv.org/pdf/1904.05673.pdf) <kbd>CVPR 2019</kbd> (Real Time, Low Power)
- [LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://arxiv.org/pdf/1904.03000.pdf) <kbd>CVPR 2019</kbd>
- [L3-Net: Towards Learning based LiDAR Localization for Autonomous Driving](https://songshiyu01.github.io/pdf/L3Net_W.Lu_Y.Zhou_S.Song_CVPR2019.pdf) <kbd>CVPR 2019</kbd>
- [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](https://arxiv.org/pdf/1903.10955.pdf) <kbd>CVPR 2019</kbd> (@SenseTime)
- [Sparse and Dense Data with CNNs: Depth Completion and Semantic Segmentation](https://arxiv.org/pdf/1808.00769.pdf) <kbd>3DV 2018</kbd>
- [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/pdf/1406.2283.pdf) <kbd>NIPS 2014</kbd> (Eigen et al)
- [3D Deep Learning Tutorial at CVPR 2017](https://www.youtube.com/watch?v=8CenT_4HWyY) [[Notes](paper_notes/3ddl_cvpr2017.md)] - (WIP)
- [Review of Graph Spectrum Theory](paper_notes/graph_spectrum.md) (WIP)
- [Learning Depth from Monocular Videos using Direct Methods](https://arxiv.org/abs/1712.00175) <kbd>CVPR 2018</kbd> (monocular depth estimation)
- [Polar Transformer Networks](https://arxiv.org/abs/1709.01889) <kbd>ICLR 2018</kbd>
- [Circular Object Detection in Polar Coordinates for 2D LIDAR Data](https://www.researchgate.net/publication/309365539_Circular_Object_Detection_in_Polar_Coordinates_for_2D_LIDAR_DataCCPR2016) <kbd>CCPR 2016</kbd>

## Technical Debt
- CRF
- [Visual SLAM and Visual Odometry](https://link.springer.com/content/pdf/10.1007%2Fs40903-015-0032-7.pdf)
- ORB SLAM
- ORB (ICCV 2011)
- Bundle Adjustment
- 3D vision
- Codebase of STN
- Codebase of monodepth
- Codebase of KITTI devkit