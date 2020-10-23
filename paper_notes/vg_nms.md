# [Visibility Guided NMS: Efficient Boosting of Amodal Object Detection in Crowded Traffic Scenes](https://ml4ad.github.io/files/papers/Visibility%20Guided%20NMS:%20Efficient%20Boosting%20of%20Amodal%20Object%20Detection%20in%20Crowded%20Traffic%20Scenes.pdf)

_June 2020_

tl;dr: Use modal bbox to guide NMS of amodal bbox.

#### Overall impression
The paper addressed a critical issue for autonomous driving in parking lot and in urban areas where many parked cars are heavily occluded. Same issue happens to crowds of pedestrians as well.

Humans perceive the world by predicting the bbox covering the entire object even if it is occluded. This is called amodal perception. (cf. [Amodal completion](amodal_completion.md))

This is very similar to [R2 NMS](r2_nms.md) in CVPR 2020, which focuses on crowd pedestrian detection.

#### Key ideas
- Training object detector with 4 additional attributes. Thus it predicts both the visible part (pixel-based bbox) and the entire object (amodal bbox).
- VG-NMS: NMS is performed on the pixel-based bbox that describe the actually visible parts but output the amodal bboxes that belong to the indices that rare retained during pixel-based NMS.
- Pixel based modal bbox can be generated from segmentation mask or ordered amodal bbox. 

#### Technical details
- **don't care** objects: KITTI ignore 25x25 pixels, and cityscape ignore 10x10 pixels. 
- VG-NMS is better than soft NMS. Soft NMS does not seem to improve performance much over NMS. 
- Simultaneously regressing amodal and modal bboxes leads to better performance, under standard NMS. 

#### Notes
- The odal bbox (pixel bbox) used in VG-NMS can be derived from amodal bbox by sorting orders and mark the non-occluded part. 

