# [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199)

_August 2019_

tl;dr: Detect 2D oriented bbox with BEV maps by adding angle regression to YOLO.

#### Overall impression
The paper is clearly written and the innovation is limited. However the performance is really nice -- this is exactly the type of paper industry likes. 

It is twice slower than [Point Pillars](point_pillars.md) achieves 115 fps.

#### Key ideas
- Add angle regression to YOLO.
- IoU calculation is updated to accommodate oriented bbox.
- The input encoding is based on MV3D. 
- Each grid has only five anchor bboxes with different headings. **The anchors do not cover a full grid but rather a finite combination of the parameters.** 
- Angle loss only effective when the oriented bbox IOU is larger than a threshold.
- Almost 10 times faster than VoxelNet, at 50 fps. In comparison [Point Pillars](point_pillars.md) achieves 115 fps.

#### Technical details
- FOV is 40 m x 80 m (same with radar). The image format is 512 x 1024.
- RGB map encoded by height, intensity and density. 
- The camera FOV is only about 90 (similar to radar). The heatmap of GT is very helpful. Output outside FOV is filtered before evaluation.
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWHbunqAYDwWz4VFroMv0QoORP6RA0hgg0Ck11Ar3F_B43OAELvA)

#### Notes
- Github repos of unofficial implementations: [here](https://github.com/AI-liu/Complex-YOLO) and [here with uncertainty](https://github.com/wl5/complex_yolo_3d) and [here](https://github.com/ghimiredhikura/Complex-YOLOv3)

