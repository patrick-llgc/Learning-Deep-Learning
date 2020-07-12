# [WiderPerson: A Diverse Dataset for Dense Pedestrian Detection in the Wild](https://arxiv.org/abs/1909.12118)

_July 2020_

tl;dr: A relatively scale (8k training images) dataset for crowded/dense human detection.

#### Overall impression
Overall not quite impressive. It fails to cite a closely related dataset [CrowdHuman](crowdhuman.md), and ablation study of the issue is not as extensive as well.

#### Key ideas
- 30 persons per image.
- Annotate top of the head and middle of the feet (similar to CityPerson). The bbox is automatically generated with aspect ratio of 0.41. This is 
- Difficulty: > 100 pixel (easy), > 50 pixel (medium), > 20 pixel (hard). Similar to WiderFace.
- NMS is a problem in crowded scenes, but it is not handled in this paper. Maybe try [Visibility Guided NMS](vg_nms.md).

#### Technical details
- Use pHash to avoid duplication of images. 
- Annotation tool with examples in the GUI.
	- ![](https://cdn-images-1.medium.com/max/1600/1*9SyZeiUg-sjrrwZsFa7FIA.png)
- Evaluation metric: MR

#### Notes
- [Tsinghua-Daimler datasets for cyclists](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Tsinghua-Daimler_Cyclist_Detec/tsinghua-daimler_cyclist_detec.html)
	- Bounding Box based labels are provided for the classes: ("pedestrian", "cyclist", "motorcyclist", "tricyclist", "wheelchairuser", "mopedrider").
- [The EuroCity Persons Dataset: A Novel Benchmark for Object Detection](https://arxiv.org/abs/1805.07193) <kbd>T-PAMI 2019</kbd>