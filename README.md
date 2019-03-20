# Paper notes
This repository contains my paper reading notes on deep learning and machine learning. **New year resolution for 2019: read at least one paper a week!**

## Summary of Topics

| Topics                                                  | Description                 |
| ------------------------------------------------------------ | --------------------------- |
| <kbd>DRL</kbd>                                               | Deep Reinforcement Learning |
| <kbd>CLS</kbd>                                                | Classification               |
| <kbd>OD</kbd>                                                | Object Detection            |
| <kbd>InsSeg</kbd>, <kbd>SemSeg</kbd>, <kbd>PanSeg</kbd> | Instance/Semantic/Panoptic Segmentation |
| <kbd>Video</kbd> | Video understanding |
| <kbd>MI</kbd> | Medical Imaging |
| <kbd>GAN</kbd> | Generative Adversarial Network |
| <kbd>NIPS</kbd>, <kbd>CVPR</kbd>, <kbd>ICCV</kbd>, <kbd>ECCV</kbd>, etc | Conference papers           |

## 2019-04
- [Multi-Task Multi-Sensor Fusion for 3D Object Detection]() (TBD from Uber CVPR 2019)
- [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://arxiv.org/pdf/1712.02294.pdf) (AVOD) <kbd>IROS 2018</kbd> (birds eye view)
- [DeLS-3D: Deep Localization and Segmentation with a 3D Semantic Map
](https://arxiv.org/pdf/1805.04949.pdf) <kbd>CVPR 2018</kbd>
- [The ApolloScape Open Dataset for Autonomous Driving and its Application](https://arxiv.org/pdf/1803.06184.pdf) (dataset, point cloud)
- [2D-Driven 3D Object Detection in RGB-D Images](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lahoud_2D-Driven_3D_Object_ICCV_2017_paper.pdf) <kbd>ICCV 2017</kbd>
- [3D Deep Learning Tutorial at CVPR 2017](https://www.youtube.com/watch?v=8CenT_4HWyY) [[Notes](paper_notes/3ddl_cvpr2017.md)] - (WIP)
- [A Multi-Sensor Fusion System for Moving Object Detection and Tracking in Urban Driving Environments](http://www.cs.cmu.edu/~youngwoo/doc/icra-14-sensor-fusion.pdf) <kbd>ICRA 2014</kbd>
- [Intro：Sensor Fusion for Adas 无人驾驶中的数据融合 (from 知乎)](https://zhuanlan.zhihu.com/p/40967227) (Up to CVPR 2018)
- [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/pdf/1812.07179.pdf) <kbd>CVPR 2019</kbd>
- [Deep Continuous Fusion for Multi-Sensor 3D Object Detection](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf) <kbd>ECCV 2018</kbd> (sensor fusion, birds eye view)
- [PIXOR: Real-time 3D Object Detection from Point Clouds](https://arxiv.org/pdf/1902.06326.pdf) <kbd>CVPR 2018</kbd> (birds eye view)
- [PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation](https://arxiv.org/pdf/1807.00652.pdf) (pointnet alternative, backbone)


## 2019-03
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf) [[Notes](paper_notes/bag_of_freebies_object_detection.md)]
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf) [[Notes](paper_notes/mixup.md)] <kbd>ICLR 2018</kbd>
- [Multi-view Convolutional Neural Networks for 3D Shape Recognition](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.pdf) (MVCNN) [[Notes](paper_notes/mvcnn.md)] <kbd>ICCV 2015</kbd> 
- [3D ShapeNets: A Deep Representation for Volumetric Shapes](http://3dshapenets.cs.princeton.edu/paper.pdf) [[Notes](paper_notes/3d_shapenets.md)] <kbd>CVPR 2015</kbd>
- [Volumetric and Multi-View CNNs for Object Classification on 3D Data](https://arxiv.org/pdf/1604.03265.pdf) [[Notes](paper_notes/vol_vs_mvcnn.md)] <kbd>CVPR 2016</kbd>
- [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf) [[Notes](paper_notes/groupnorm.md)] <kbd>ECCV 2018</kbd>
- [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf) [[Notes](paper_notes/stn.md)] <kbd>NIPS 2015</kbd>
- [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/pdf/1711.08488.pdf) (F-PointNet) [[Notes](paper_notes/frustum_pointnet.md)] <kbd>CVPR 2018</kbd> 
- [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/pdf/1801.07829.pdf) [[Notes](paper_notes/edgeconv.md)]
- [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/pdf/1812.04244v1.pdf) (SOTA for 3D object detection) [[Notes](paper_notes/point_rcnn.md)]
- [Multi-View 3D Object Detection Network for Autonomous Driving](https://arxiv.org/pdf/1611.07759.pdf) (MV3D) [[Notes](paper_notes/mv3d.md)] <kbd>CVPR 2017</kbd> (sensor fusion)



## 2019-02
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf) (I3D) [[Notes](paper_notes/quo_vadis_i3d.md)]<kbd>Video</kbd> <kbd>CVPR 2017</kbd>
- [Initialization Strategies of Spatio-Temporal Convolutional Neural Networks](https://arxiv.org/pdf/1503.07274.pdf) [[Notes](paper_notes/quo_vadis_i3d.md)] <kbd>Video</kbd>
- [Detect-and-Track: Efficient Pose Estimation in Videos](https://arxiv.org/pdf/1712.09184.pdf) [[Notes](paper_notes/quo_vadis_i3d.md)] <kbd>ICCV 2017</kbd> <kbd>Video</kbd>
- [Deep Learning Based Rib Centerline Extraction and Labeling](https://arxiv.org/pdf/1809.07082) [[Notes](paper_notes/rib_centerline_philips.md)] <kbd>MI</kbd> <kbd>MICCAI 2018</kbd>
- [SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf) [[Notes](paper_notes/slowfast.md)] <kbd>Video</kbd>
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) (ResNeXt) [[Notes](paper_notes/resnext.md)] <kbd>CVPR 2017</kbd>.
- [Beyond the pixel plane: sensing and learning in 3D](https://thegradient.pub/beyond-the-pixel-plane-sensing-and-learning-in-3d/) (blog, [中文版本](https://zhuanlan.zhihu.com/p/44386618))
- [VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf) (VoxNet) [[Notes](paper_notes/voxnet.md)]
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf) <kbd>CVPR 2017</kbd> [[Notes](paper_notes/pointnet.md)]
- [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf) <kbd>NIPS 2017</kbd> [[Notes](paper_notes/pointnet++.md)]
- [Review of Geometric deep learning 几何深度学习前沿 (from 知乎)](https://zhuanlan.zhihu.com/p/36888114) (Up to CVPR 2018)
- [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/pdf/1711.06396.pdf) <kbd>CVPR 2018</kbd> 


## 2019-01
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
- [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
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
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) (MobileNet)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf) (MobileNet v2)
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) (Xception)
- [SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving](https://arxiv.org/pdf/1612.01051.pdf) (SqueezeDet)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf)
- [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://openreview.net/forum?id=Bygh9j09KX) <kbd>ICML 2019</kbd>
- [Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet](https://openreview.net/forum?id=SkfMWhAqYQ) (BagNet) [blog](https://blog.evjang.com/2019/02/bagnet.html) <kbd>ICML 2019</kbd>
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820v2.pdf)
- [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf)


### 3D Object detection
Estimates oriented 3D bounding boxes.

- [IPOD: Intensive Point-based Object Detector for Point Cloud](https://arxiv.org/pdf/1812.05276.pdf)

Treating RGBD data as diff channels of 2D images

- [Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images](https://cis.temple.edu/~latecki/Papers/DengCVPR2017.pdf) <kbd>CVPR 2017</kbd>
- [Vehicle Detection from 3D Lidar Using Fully Convolutional Network](https://arxiv.org/pdf/1608.07916.pdf)
- [2D-Driven 3D Object Detection in RGB-D Images](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lahoud_2D-Driven_3D_Object_ICCV_2017_paper.pdf)
- [3D-SSD: Learning Hierarchical Features from RGB-D Images for Amodal 3D Object Detection](https://arxiv.org/pdf/1711.00238.pdf)

### 2D Object detection
- [Mask Scoring R-CNN](https://arxiv.org/pdf/1903.00241.pdf) <kbd>CVPR 2019</kbd>
- [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/pdf/1604.03540.pdf)

### Instance and Panoptic Segmentation
- [Path Aggregation Network for Instance Segmentation](https://arxiv.org/pdf/1803.01534.pdf)

### Video Understanding
- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767.pdf) (C3D)  <kbd> Video </kbd><kbd> ICCV 2015 </kbd>
- [AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions](https://arxiv.org/pdf/1705.08421.pdf)
- [Spatiotemporal Residual Networks for Video Action Recognition](https://arxiv.org/pdf/1611.02155.pdf) (decouple spatiotemporal) <kbd>NIPS 2016</kbd>
- [Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks](https://arxiv.org/pdf/1711.10305.pdf) (P3D, decouple spatiotemporal) <kbd> ICCV 2017 </kbd>
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/pdf/1711.11248.pdf) (decouple spatiotemporal) <kbd>CVPR 2018</kbd>
- [Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://arxiv.org/pdf/1712.04851.pdf) (decouple spatiotemporal) <kbd>ECCV 2018</kbd>
- [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/pdf/1711.09577.pdf) <kbd>CVPR 2018</kbd>

### Medical Imaging
- [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/pdf/1809.10486.pdf)
- [Foveal fully convolutional nets for multi-organ segmentation](https://www.dropbox.com/s/zcmqheb0cbwsxq0/Foveal%20fully%20convolutional%20nets%20for%20multi-organ%20segmentation.pdf?dl=0)
- [Evaluation of deep learning methods for parotid gland segmentation from CT images](https://www.spiedigitallibrary.org/journals/Journal-of-Medical-Imaging/volume-6/issue-1/011005/Evaluation-of-deep-learning-methods-for-parotid-gland-segmentation-from/10.1117/1.JMI.6.1.011005.pdf)
- [DeepMedic for Brain Tumor Segmentation](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/kamnitsas2016brats.pdf)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999.pdf)
- [Attention-Gated Networks for Improving Ultrasound Scan Plane Detection](https://arxiv.org/pdf/1804.05338.pdf)
- [3D Context Enhanced Region-based Convolutional Neural Network for End-to-End Lesion Detection](https://arxiv.org/pdf/1806.09648.pdf)

### Orthoganal architecture improvements
- [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/pdf/1803.02579.pdf)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521.pdf)

### GAN

### Reinforcement Learning
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) <kbd> NIPS 2013 </kbd>
- [Multi-Scale Deep Reinforcement Learning for Real-Time 3D-Landmark Detection in CT Scan](http://comaniciu.net/Papers/MultiscaleDeepReinforcementLearning_PAMI18.pdf)
- [An Artificial Agent for Robust Image Registration](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14751/14296)

### 3D Perception
- [** 3D-CNN：3D Convolutional Neural Networks for Landing Zone Detection from LiDAR**](https://www.ri.cmu.edu/pub_files/2015/3/maturana-root.pdf)
- [Generative and Discriminative Voxel Modeling with Convolutional Neural Networks](https://arxiv.org/pdf/1608.04236.pdf)
- [Orientation-boosted Voxel Nets for 3D Object Recognition](https://arxiv.org/pdf/1604.03351.pdf) (ORION) <BMVC 2017>
- [GIFT: A Real-time and Scalable 3D Shape Search Engine](https://arxiv.org/pdf/1604.01879.pdf) <kbd>CVPR 2016</kbd>
- [3D Shape Segmentation with Projective Convolutional Networks](https://people.cs.umass.edu/~kalo/papers/shapepfcn/) (ShapePFCN)<kbd>CVPR 2017</kbd>
- [Learning Local Shape Descriptors from Part Correspondences With Multi-view Convolutional Networks](https://arxiv.org/pdf/1706.04496.pdf)
- [Open3D: A Modern Library for 3D Data Processing](http://www.open3d.org/wordpress/wp-content/paper.pdf)
- [Multimodal Deep Learning for Robust RGB-D Object Recognition](https://arxiv.org/pdf/1507.06821.pdf) <kbd>IROS 2015</kbd>
- [** FlowNet3D: Learning Scene Flow in 3D Point Clouds](https://arxiv.org/pdf/1806.01411.pdf) <kbd>CVPR 2019</kbd>
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
- [PointCNN: Convolution On X-Transformed Points](https://arxiv.org/pdf/1801.07791.pdf) <kbd>NIPS 2018</kbd>
- [** Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis](https://arxiv.org/pdf/1612.00101.pdf) <kbd>CVPR 2017</kbd>
