# [OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception](https://arxiv.org/abs/2303.03991)

_March 2023_

tl;dr: A new benchmark of NuScenes-Occupancy, with new KPI and simple baselines. 

#### Overall impression
The paper used an Augmenting And Purifying (AAP) pipeline for autolabel. This essentially leverages a functional (yet not performing) pretrained baseline model to prelabel. Human labelers are then asked to purify the pre-label. This way it can greatly boost labeling efficiency.

This pipeline aims to provide dense occupancy labels, with semantic information. It invests 4000 human hours to refine the label.

The idea of Cascade Occupancy Network (CONet) to improve latency is relevant to industrial deployment. The simple idea of bilinear interpolation seems to work quite well already.

The pipeline to generate occupancy label is Poisson Recon used by [SurroundOcc](surroundocc.md), and Augment-and-Purify (AAP) pipeline proposed by [OpenOccupancy](openoccupancy.md). They share the initial steps but are different in the refinement step. It would be interesting to see a side-by-side comparison of the two.

#### Key ideas
- The semantic occupancy perception has two tasks: 1) volumetric occupancy and 2) semantic labels.
- AAP (Augmenting And Purifying) pipeline
	- V_init semantic labels are from aggregated lidar semantic segmentation. The aggregation process is done separately for dynamic objects and static scenes, similar to that used in [SurroundDepth](surrounddepth.md).
	- With V_init, a baseline model F_m is trained. The prediction result is V_pseudo.
	- V_aug is a fusion between V_init and V_pseudo. Specifically, V_pesudo augments V_init, but never overwrites V_init.
- Evaluation benchmark --> This is pretty much the same as the original KPI proposed for SSC.
	- IoU: geometric metric, empty or occupied, binary
	- mIoU: mean IoU of each class, following lidar semantic segmentation
	- "noise" is ignored in evaluaton. ("For bounding boxes that overlap, we resolve them by labeling the overlapping points as noise. ...less than 0.8%", from [Panoptic nuScenes](https://arxiv.org/abs/2109.03805))

#### Technical details
- Resolution is 0.2 m. Range is 40x512x512. Roughly **50 meters** to all sides. --> Same resolution as [SemanticKITTI](semantickiti.md).
- Two drawbacks of SemanticKITTI
	- it lacks diversity in urban scenes, which hinders the generalization
	- only evaluates front-view occupancy predictions.
- Baseline model
	- [BEVFusion](bevfusion.md), following [LSS](lift_splat_shoot.md).
	- Stride parameter is set to 4 by default, but too coarse.
- CONet (Cascade Occupancy Network)
	- The coarse-to-fine pipeline includes with the **semantic feature** (by casting back to images and collect image features) and **geometric features** (by bilinear interpolation). --> Ablation study shows that with geometric features alone, the performance is quite good. Semantic features helps very little.
- The improvement of OCNet baseline over [TPVFormer](tpvformer.md) should stem from the fact that OCNet is densely supervised. 

#### Notes
- The geometric metric only classifies a voxel into occupied or empty. --> This is too coarse. We should include at least differentiate "occluded" and "void". [CPVR 2023 challenge](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction#evaluation-metrics) seems to have occluded as one semantic class.