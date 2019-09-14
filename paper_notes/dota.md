# [DOTA: A Large-scale Dataset for Object Detection in Aerial Images](https://vision.cornell.edu/se3/wp-content/uploads/2018/03/2666.pdf)

_September 2019_

tl;dr: Largest dataset for object detection in aerial images.

#### Overall impression
This dataset addresses a specific field called Earth Vision/Earth Observation/Remote Sensing.

In aerial images, there is rarely occlusion so every object can be annotated (vs. crowd class in COCO dataset).

Other than aerial image, text region detection also involves oriented bbox detection.

#### Key ideas
- The annotation is 8 dof quadrilateral. But essentially most of them are (or converted to) Oriented bounding box (OBB).
- For horizontal bounding boxes, sometimes the overlap is too big for object detection algorithms to tell them apart (due to NMS).
- Cars: Big car (trucks, etc) and small car two categories.

#### Technical details
- Dataset stats are analyzed to filter anomaly annotations.

#### Notes
- According to their implementation of the Faster RCNN (OBB), they used original anchor proposals, and reparameterized the anchor box to four corners (8 points), and then changed prediction from 4 numbers to 8 numbers. No oriented anchors were used. --> compare with [RoiTransformer](roi_transformer.md).
