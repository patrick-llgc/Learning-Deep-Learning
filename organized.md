
# Organized paper notes
Last Updated: 07/04/2019

The following organizes the paper reading notes according to subfields. 

## Deep Learning in General
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf) [[Notes](paper_notes/bag_of_tricks_cnn.md)] <kbd>CLS</kbd>
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf) [[Notes](paper_notes/bag_of_freebies_object_detection.md)]
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf) [[Notes](paper_notes/mixup.md)] <kbd>ICLR 2018</kbd>
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) (ResNeXt) [[Notes](paper_notes/resnext.md)] <kbd>CVPR 2017</kbd>
- [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) [[Notes](paper_notes/groupnorm.md)] <kbd>ECCV 2018</kbd>
- [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf) [[Notes](paper_notes/stn.md)] <kbd>NIPS 2015</kbd>

### Datasets
- [Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf) [[Notes](paper_notes/kitti.md)] <kbd>CVPR 2012</kbd>
- [Vision meets Robotics: The KITTI Dataset](http://ww.cvlibs.net/publications/Geiger2013IJRR.pdf) [[Notes](paper_notes/kitti.md)] <kbd>IJRR 2013</kbd>
- [A New Performance Measure and Evaluation Benchmark
for Road Detection Algorithms](http://www.cvlibs.net/publications/Fritsch2013ITSC.pdf) [[Notes](paper_notes/kitti_lane.md)] <kbd>ITSC 2013</kbd>

### Bayesian DL
- [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/pdf/1703.04977.pdf) [[Notes](paper_notes/uncertainty_bdl.md)] <kbd>NIPS 2017</kbd>
- [Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding](https://arxiv.org/pdf/1511.02680.pdf) [[Notes](paper_notes/bayesian_segnet.md)]<kbd>BMVC 2017</kbd>

### DRL
- [Human-level control through deep reinforcement learning (Nature DQN paper)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) [[Notes](paper_notes/nature_dqn_paper.md)] <kbd>DRL</kbd>
- [Deep Reinforcement Learning for Vessel Centerline Tracing in Multi-modality 3D Volumes](https://link.springer.com/chapter/10.1007/978-3-030-00937-3_86) [[Notes](paper_notes/drl_vessel_centerline.md)] <kbd>DRL</kbd> <kbd>MI</kbd>
- [Deep Reinforcement Learning for Flappy Bird](http://cs229.stanford.edu/proj2015/362_report.pdf) [[Notes](paper_notes/drl_flappy.md)] <kbd>DRL</kbd>


## 2D Object Detection
- [Optimizing the Trade-off between Single-Stage and Two-Stage Object Detectors using Image Difficulty Prediction](https://arxiv.org/pdf/1803.08707.pdf) (Very nice illustration of 1 and 2 stage object detection)
- [Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://arxiv.org/pdf/1711.07264.pdf) (Megvii) [[Notes](paper_notes/lighthead_rcnn.md)]
- [Object Detection based on Region Decomposition and Assembly](https://arxiv.org/pdf/1901.08225.pdf) [[Notes](paper_notes/object_detection_region_decomposition.md)] <kbd> AAAI 2019 </kbd>
- [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/abs/1811.04533) [[Notes](paper_notes/m2det.md)] <kbd> AAAI 2019 </kbd>

### Anchor-free
- Review of Anchor-free methods (知乎Blog) [目标检测：Anchor-Free时代](https://zhuanlan.zhihu.com/p/62103812) [Anchor free深度学习的目标检测方法](https://zhuanlan.zhihu.com/p/64563186) [My Slides on CSP](https://docs.google.com/presentation/d/1_dUfxv63108bZXUnVYPIOAdEIkRZw5BR9-rOp-Ni0X0/)
- [DenseBox: Unifying Landmark Localization with End to End Object Detection](https://arxiv.org/pdf/1509.04874.pdf)
- [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/pdf/1808.01244.pdf) [[Notes](paper_notes/cornernet.md)] <kbd>ECCV 2018</kbd>
- [ExtremeNet: Bottom-up Object Detection by Grouping Extreme and Center Points](https://arxiv.org/pdf/1901.08043.pdf) [[Notes](paper_notes/extremenet.md)] <kbd>CVPR 2019</kbd>
- [CSP: High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection](https://arxiv.org/pdf/1904.02948.pdf) (center and scale prediction) [[Notes](paper_notes/csp_pedestrian.md)] <kbd>CVPR 2019</kbd> 
- [FSAF: Feature Selective Anchor-Free Module for Single-Shot Object Detection](https://arxiv.org/pdf/1903.00621.pdf) [[Notes](paper_notes/fsaf_detection.md)] <kbd>CVPR 2019</kbd>
- [FoveaBox: Beyond Anchor-based Object Detector](https://arxiv.org/pdf/1904.03797v1.pdf) (anchor-free) [[Notes](paper_notes/foveabox.md)]
- [CenterNet: Objects as points](https://arxiv.org/pdf/1904.07850.pdf) (from ExtremeNet authors) [[Notes](paper_notes/centernet_ut.md)]
- [CenterNet: Object Detection with Keypoint Triplets](https://arxiv.org/pdf/1904.08189.pdf) [[Notes](paper_notes/centernet_cas.md)]
- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf) [[Notes](paper_notes/fcos.md)]

## Image Segmentation
- [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf) [[Notes](paper_notes/panoptic_segmentation.md)] <kbd>PanSeg</kbd>
- [Panoptic Feature Pyramid Networks](https://arxiv.org/pdf/1901.02446.pdf) [[Notes](paper_notes/panoptic_fpn.md)] <kbd>PanSeg</kbd> 
- [Attention-guided Unified Network for Panoptic Segmentation](https://arxiv.org/pdf/1812.03904.pdf) [[Notes](paper_notes/AUNet_panoptic.md)] <kbd>PanSeg</kbd>
- [Path Aggregation Network for Instance Segmentation](https://arxiv.org/pdf/1803.01534.pdf) [[Notes](paper_notes/panet.md)] <kbd> CVPR 2018 </kbd>

## 3D Object Detection
### Classification on Point Cloud
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf) <kbd>CVPR 2017</kbd> [[Notes](paper_notes/pointnet.md)]
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf) <kbd>NIPS 2017</kbd> [[Notes](paper_notes/pointnet++.md)]
- [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/pdf/1801.07829.pdf) [[Notes](paper_notes/edgeconv.md)]
- [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf) (VoxNet) [[Notes](paper_notes/voxnet.md)]
- [Multi-view Convolutional Neural Networks for 3D Shape Recognition](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf) (MVCNN) [[Notes](paper_notes/mvcnn.md)] <kbd>ICCV 2015</kbd> 
- [3D ShapeNets: A Deep Representation for Volumetric Shapes](http://3dshapenets.cs.princeton.edu/paper.pdf) [[Notes](paper_notes/3d_shapenets.md)] <kbd>CVPR 2015</kbd>
- [Volumetric and Multi-View CNNs for Object Classification on 3D Data](https://arxiv.org/pdf/1604.03265.pdf) [[Notes](paper_notes/vol_vs_mvcnn.md)] <kbd>CVPR 2016</kbd>
- [Beyond the pixel plane: sensing and learning in 3D](https://thegradient.pub/beyond-the-pixel-plane-sensing-and-learning-in-3d/) (blog, [中文版本](https://zhuanlan.zhihu.com/p/44386618))
- [Review of Geometric deep learning 几何深度学习前沿 (from 知乎)](https://zhuanlan.zhihu.com/p/36888114) (Up to CVPR 2018)

### Detection on Point Cloud
- [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf) (F-PointNet) [[Notes](paper_notes/frustum_pointnet.md)] <kbd>CVPR 2018</kbd> 
- [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/pdf/1812.04244v1.pdf) (SOTA for 3D object detection) [[Notes](paper_notes/point_rcnn.md)] <kbd>CVPR 2019</kbd>
- [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/pdf/1711.06396.pdf) <kbd>CVPR 2018</kbd> (Apple, first end-to-end point cloud encoding to grid)
- [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337/pdf) <kbd>Sensors 2018</kbd> (builds on VoxelNet)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/pdf/1812.05784.pdf) [[Notes](paper_notes/point_pillars.md)] <kbd>CVPR 2019</kbd> (builds on SECOND)

### Sensor Fusion
- [Multi-View 3D Object Detection Network for Autonomous Driving](https://arxiv.org/pdf/1611.07759.pdf) (MV3D) [[Notes](paper_notes/mv3d.md)] <kbd>CVPR 2017</kbd> (Baidu, sensor fusion, BV proposal)
- [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/pdf/1712.02294.pdf) (AVOD) [[Notes](paper_notes/avod.md)] <kbd>IROS 2018</kbd> (sensor fusion, multiview proposal)
- [Multi-Task Multi-Sensor Fusion for 3D Object Detection](http://www.cs.toronto.edu/~byang/papers/mmf.pdf) [[Notes](paper_notes/mmf.md)] <kbd>CVPR 2019</kbd> (@Uber, sensor fusion)



## Monocular 3D Object Detection

### Representation Transformation (Pseudo-Lidar, BEV)
- [Multi-Level Fusion based 3D Object Detection from Monocular Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Multi-Level_Fusion_Based_CVPR_2018_paper.pdf) [[Notes](paper_notes/mlf.md)] <kbd>CVPR 2018</kbd> (precursor to pseudo-lidar)
- [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/1812.07179.pdf) [[Notes](paper_notes/pseudo_lidar.md)] <kbd>CVPR 2019</kbd>
- [Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving](https://arxiv.org/pdf/1906.06310.pdf) [[Notes](paper_notes/pseudo_lidar++.md)]
- [Pseudo lidar-e2e: Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud](https://arxiv.org/abs/1903.09847) [[Notes](paper_notes/pseudo_lidar_e2e.md)] <kbd>ICCV 2019</kbd>
- [pseudo lidar color: Accurate Monocular 3D Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving](https://arxiv.org/abs/1903.11444) \[[Notes](paper_notes/pseudo_lidar_color.md)] <kbd>ICCV 2019</kbd>
- [ForeSeE: Task-Aware Monocular Depth Estimation for 3D Object Detection](https://arxiv.org/abs/1909.07701) \[[Notes](paper_notes/foresee_mono3dod.md)] (successor to pseudo-lidar) (mono 3DOD SOTA)
- [BEV-IPM: Deep Learning based Vehicle Position and Orientation Estimation via Inverse Perspective Mapping Image](https://ieeexplore.ieee.org/abstract/document/8814050) [[Notes](paper_notes/bev_od_ipm.md)] <kbd>IV 2019</kbd>
- [OFT: Orthographic Feature Transform for Monocular 3D Object Detection](https://arxiv.org/abs/1811.08188) \[[Notes](paper_notes/oft.md)] (Convert camera to BEV, Alex Kendall) <kbd>BMVC 2019</kbd>
- [BirdGAN: Learning 2D to 3D Lifting for Object Detection in 3D for Autonomous Vehicles](https://arxiv.org/abs/1904.08494) [[Notes](paper_notes/birdgan.md)] <kbd>IROS 2019</kbd> 


### Keypoints and Shape
- [Deep MANTA: A Coarse-to-fine Many-Task Network for joint 2D and 3D vehicle analysis from monocular image](https://arxiv.org/abs/1703.07570) [[Notes](paper_notes/deep_manta.md)] <kbd>CVPR 2017</kbd>
- [3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kundu_3D-RCNN_Instance-Level_3D_CVPR_2018_paper.pdf) [[Notes](paper_notes/3d_rcnn.md)] <kbd>CVPR 2018</kbd>
- [ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape](https://arxiv.org/abs/1812.02781) [[Notes](paper_notes/roi10d.md)] <kbd>CVPR 2019</kbd>
- [MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization](https://arxiv.org/abs/1811.10247) [[Notes](paper_notes/monogrnet.md)] <kbd>AAAI 2019</kbd> 
- [MonoGRNet 2: Monocular 3D Object Detection via Geometric Reasoning on Keypoints](https://arxiv.org/abs/1905.05618) [[Notes](paper_notes/monogrnet_russian.md)] \(estimates depth from keypoints)
- [MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation](https://arxiv.org/abs/1906.06059) [[Notes](paper_notes/monoloco.md)] <kbd>ICCV 2019</kbd>
- [GPP: Ground Plane Polling for 6DoF Pose Estimation of Objects on the Road](https://arxiv.org/abs/1811.06666) \[[Notes](paper_notes/gpp.md)] (UCSD, mono 3DOD)


### Distance via 2D/3D constraint
- [Deep3dBox: 3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496) \[[Notes](paper_notes/deep3dbox.md)] (from Zoox) <kbd>CVPR 2017</kbd>
- [Shift R-CNN: Deep Monocular 3D Object Detection with Closed-Form Geometric Constraints](https://arxiv.org/abs/1905.09970) [[Notes](paper_notes/shift_rcnn.md)] <kbd>IEEE ICIP 2019</kbd>
- [FQNet: Deep Fitting Degree Scoring Network for Monocular 3D Object Detection](https://arxiv.org/abs/1904.12681) [[Notes](paper_notes/fqnet.md)] <kbd>CVPR 2019</kbd> (Mono 3DOD, Jiwen Lu)
- [Mono3D++: Monocular 3D Vehicle Detection with Two-Scale 3D Hypotheses and Task Priors](https://arxiv.org/abs/1901.03446) [[Notes](paper_notes/mono3d++.md)] (Stefano Soatto) <kbd>AAAI 2019</kbd>
- [GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving](https://arxiv.org/abs/1903.10955) [[Notes](paper_notes/gs3d.md)] <kbd>CVPR 2019</kbd> (2d bbox height)
- [MonoPSR: Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction](https://arxiv.org/abs/1904.01690) [[Notes](paper_notes/monopsr.md)] <kbd>CVPR 2019</kbd> (2d bbox height)
- [CasGeo: 3D Bounding Box Estimation for Autonomous Vehicles by Cascaded Geometric Constraints and Depurated 2D Detections Using 3D Results](https://arxiv.org/abs/1909.01867) [[Notes](paper_notes/casgeo.md)]
- [MVRA: Multi-View Reprojection Architecture for Orientation Estimation](http://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Choi_Multi-View_Reprojection_Architecture_for_Orientation_Estimation_ICCVW_2019_paper.pdf) [[Notes](paper_notes/mvra.md)] <kbd>ICCV 2019</kbd>


### Direct generation of 3D proposals
- [Mono3D: Monocular 3D Object Detection for Autonomous Driving](https://www.cs.toronto.edu/~urtasun/publications/chen_etal_cvpr16.pdf) [[Notes](paper_notes/mono3d.md)] <kbd>CVPR2016</kbd> (lots of 3D anchors)
- [TLNet: Triangulation Learning Network: from Monocular to Stereo 3D Object Detection](https://arxiv.org/abs/1906.01193) [[Notes](paper_notes/tlnet.md)] <kbd>CVPR 2019</kbd> (mono baseline with 3D anchors)
- [SS3D: Monocular 3D Object Detection and Box Fitting Trained End-to-End Using Intersection-over-Union Loss](https://arxiv.org/abs/1906.08070) [[Notes](paper_notes/ss3d.md)] \(rergess distance from images, centernet like)
- [M3D-RPN: Monocular 3D Region Proposal Network for Object Detection](https://arxiv.org/abs/1907.06038) [[Notes](paper_notes/m3d_rpn.md)] <kbd>ICCV 2019</kbd> (Xiaoming Liu)
- [MonoDIS: Disentangling Monocular 3D Object Detection](https://arxiv.org/abs/1905.12365) [[Notes](paper_notes/monodis.md)] <kbd>ICCV 2019</kbd>
- [Joint Monocular 3D Vehicle Detection and Tracking](https://arxiv.org/abs/1811.10742) [[Notes](paper_notes/mono_3d_tracking.md)] <kbd>ICCV 2019</kbd> (directly regress 1/d, Berkeley DeepDrive)
- [CenterNet: Objects as points](https://arxiv.org/pdf/1904.07850.pdf) (from ExtremeNet authors) [[Notes](paper_notes/centernet_ut.md)] (directly regress 1/d)
- [Joint Monocular 3D Vehicle Detection and Tracking](https://arxiv.org/abs/1811.10742) [[Notes](paper_notes/mono_3d_tracking.md)] <kbd>ICCV 2019</kbd> (Berkeley DeepDrive)


### Others
- [Deep Optics for Monocular Depth Estimation and 3D Object Detection](https://arxiv.org/abs/1904.08601) [[Notes](paper_notes/deep_optics.md)] <kbd>ICCV 2019</kbd>



## Depth Estimation
- [Deep Depth Completion of a Single RGB-D Image](https://arxiv.org/pdf/1803.09326.pdf) [[Notes](paper_notes/deep_depth_completion_rgbd.md)] <kbd>CVPR 2018</kbd> (indoor)
- [DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image](https://arxiv.org/pdf/1812.00488v2.pdf) [[Notes](paper_notes/deeplidar.md)] <kbd>CVPR 2019</kbd> (outdoor)
- [SfMLearner: Unsupervised Learning of Depth and Ego-Motion from Video](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf) [[Notes](paper_notes/sfm_learner.md)] <kbd>CVPR 2017</kbd>
- [Monodepth2: Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260) [[Notes](paper_notes/monodepth2.md)] \(@Niantic)

### Practical Autonomous Driving Topics
- [MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving](https://arxiv.org/pdf/1612.07695.pdf) [[Notes](paper_notes/multinet_raquel.md)]
- [TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents](https://arxiv.org/pdf/1811.02146.pdf) [[Notes](paper_notes/trafficpredict.md)] <kbd>AAAI 2019 (oral)</kbd>
- [Semantic Segmentation on Radar Point Clouds](https://ieeexplore.ieee.org/document/8455344) [[[Notes](paper_notes/radar_semantic_seg.md)]] (from Daimler AG) <kbd> FUSION 2018 </kbd>
- [DeepSignals: Predicting Intent of Drivers Through Visual Signals](https://arxiv.org/pdf/1905.01333.pdf) [[Notes](paper_notes/deep_signals.md)] <kbd>ICRA2019</kbd> (@Uber, turn signal detection)
- [Deep Radar Detector](assets/papers/deep_radar_detector.pdf) [[Notes](paper_notes/deep_radar_detector.md)] <kbd> RadarCon 2019</kbd>


## Model Compression
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) (MobileNets) [[Notes](paper_notes/mobilenets.md)]
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf) (MobileNets v2) [[Notes](paper_notes/mobilenets_v2.md)] <kbd>CVPR 2018</kbd>
- [MobileNetV3: Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) [[Notes](paper_notes/mobilenets_v3.md)]
- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf) [[Notes](mnasnet.md)] <kbd>CVPR 2019</kbd> 
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) [[Notes](paper_notes/efficientnet.md)] <kbd>ICML 2019</kbd>
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/pdf/1608.08710.pdf) [[Notes](paper_notes/pruning_filters.md)] <kbd>ICLR 2017</kbd>
- [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/pdf/1707.06168.pdf) <kbd>ICCV 2017</kbd> (Face++, Yihui He) [[Notes](paper_notes/channel_pruning_megvii.md)]
- [Layer-compensated Pruning for Resource-constrained Convolutional Neural Networks](https://arxiv.org/pdf/1810.00518.pdf) [[Notes](paper_notes/layer_compensated_pruning.md)] <kbd>NIPS 2018 Talk</kbd>
- [LeGR: Filter Pruning via Learned Global Ranking](https://arxiv.org/pdf/1904.12368.pdf) [[Notes](paper_notes/legr.md)]
- [Rethinking the Value of Network Pruning](https://arxiv.org/pdf/1810.05270.pdf) <kbd>ICLR 2019</kbd>
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) [[Notes](paper_notes/lottery_ticket_hypothesis.md)] <kbd> ICLR 2019 </kbd>


### NAS
- [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](https://arxiv.org/pdf/1904.07392.pdf) [[Notes](paper_notes/nas_fpn.md)] <kbd> CVPR 2019 </kbd>
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501) [[Notes](paper_notes/autoaugment.md)] <kbd> CVPR 2019 </kbd>
- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf) <kbd>ECCV 2018</kbd> (Song Han, Yihui He)


## Video Recognition
- [Long-Term Feature Banks for Detailed Video Understanding](https://arxiv.org/pdf/1812.05038.pdf) [[Notes](paper_notes/long_term_feat_bank.md)] <kbd>Video</kbd> 
- [Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf) [[Notes](paper_notes/non_local_net.md)] <kbd>Video</kbd> <kbd>CVPR 2018</kbd>
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf) (I3D) [[Notes](paper_notes/quo_vadis_i3d.md)]<kbd>Video</kbd> <kbd>CVPR 2017</kbd>
- [Initialization Strategies of Spatio-Temporal Convolutional Neural Networks](https://arxiv.org/pdf/1503.07274.pdf) [[Notes](paper_notes/quo_vadis_i3d.md)] <kbd>Video</kbd>
- [Detect-and-Track: Efficient Pose Estimation in Videos](https://arxiv.org/pdf/1712.09184.pdf) [[Notes](paper_notes/quo_vadis_i3d.md)] <kbd>ICCV 2017</kbd> <kbd>Video</kbd>
- [SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf) [[Notes](paper_notes/slowfast.md)] <kbd>Video</kbd>


## Medical Imaging
- [Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection](https://arxiv.org/pdf/1811.08661.pdf) [[Notes](paper_notes/retina_unet.md)] <kbd>MI</kbd>
- [Deep Learning Based Rib Centerline Extraction and Labeling](https://arxiv.org/pdf/1809.07082) [[Notes](paper_notes/rib_centerline_philips.md)] <kbd>MI</kbd> <kbd>MICCAI 2018</kbd>
