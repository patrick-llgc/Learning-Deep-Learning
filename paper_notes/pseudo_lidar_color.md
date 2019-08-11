# [Accurate Monocular 3D Object Detection via Color-Embedded 3D Reconstruction for Autonomous Driving](https://arxiv.org/pdf/1903.11444.pdf) 

_August 2019_

tl;dr: Concurrent proj with Pseudo-lidar but with color embedding. SOTA together with [CMU PL](pseudo_lidar_e2e.md) for mono 3DOD as of 08/2019.

#### Overall impression
This paper seems to be inspired by pseudo-lidar but did not mention it explicitly. The paper's writing leaves much to be desired. Performance-wise, it is very similar to [CMU's e2e pseudo-lidar paper](pseudo_lidar_e2e.md).

Simply concat D to RGB does not help much, such as [ROI 10D](roi10d.md) and [MLF](mlf.md).

The paper proposed a simple segmentation method to combat the long-tail problem, although it does not clearly state so.

#### Key ideas
- The paper has several tweaks of [F-Pointnet](frustum_pointnet.md). 
	- First a much coarser thresholding is used to replace point cloud segmentation in the frustum. --> this is used to combat the long-tail problem, as also mentioned in [CMU's end-to-end pseudo-lidar paper](pseudo_lidar_e2e.md). 
	- RGB info is used to enhance point cloud. However direct concat does not work. The paper proposed attention mechanism to fuse RGB with XYZ point info. 
- Depth prior baed point cloud segmentation: threshold by depth interval based on depth mean. It is much faster and much more robust (5 ms on CPU).

#### Technical details
- The calib file is used to lift RGB to point cloud. However this can be learned for the same camera, if we only care about one set of camera.
- The RGB fusion module is also effective for real point cloud based methods. 

#### Notes
- Looks like crude thresholding works for pseudo-point cloud segmentation --> Maybe use radar info directly correct segmented pseudo-point cloud?
- Maybe we should relax evaluation method for distant objects? 1. 3d bbox gt is labeled on sparse point cloud, 2. Not as critical as nearby objects. 

