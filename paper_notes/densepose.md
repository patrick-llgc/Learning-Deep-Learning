# [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/abs/1802.00434)

_September 2020_

tl;dr: Annotating human surface correspondence on RGB images and adapt Mask RCNN for prediction. 

#### Overall impression
The paper proposed DensePose COCO dataset, and establishes dense correspondences between an RGB image and a surface-based representation of the human body. 

#### Key ideas
- Partitioning and UV parameterization of the human body. 
	- Since the human body has a complicated structure, we break it into multiple independent pieces and parameterize each piece using a local two-dimensional coordinate system, that identifies the position of any node on this surface part, based on unwrapping of the [SMPL](https://smpl.is.tue.mpg.de/) model.
- **Sample points for annotation**: roughly equidistant points obtained via k-means and request annotators to bring these points in correspondence with the surface. Note that the sample points are selected in the image domain, not surface domain.
- **Knowledge distillation**: Only a randomly chosen subset of image pixels per training sample is annotated. These sparse correspondence is to train a teacher model and then in-paint the supervision signal on the rest of the image mask.
	- Much better results can be obtained by in-painting the values of supervision signals that were not originally annotated. --> See [Data distillation](data_distillation.md).
- Evaluation: via geodesic distance (actually euclidean distance in UV space).
- Architecture:
	- Modified Mask RCNN for this purpose
	- Regress part class (24 + 1), and then 24 regressor heads to predict (u, v).
	- Cross-cascading multitask learning: The output of one head feeds into a different task. The additional guidance from keypoint prediction branch boosts the dense pose prediction a lot. 

#### Technical details
- The user interface for 3D annotation is well designed. Once the target point is annotated, the point is displayed on all rendered images simultaneously. This avoid manually rotating the surface. 
![](https://media.arxiv-vanity.com/render-output/3760777/Figures/correspondence_interface.png)
- Accuracy of human annotators: generate images with GT by rendering human CAD models. This avoids different annotators to annotate the same object to get "consensus" GT.
	- Errors are small on small surface parts with distinctive features such as face, hands and feet. For large uniform areas the annotator errors can get larger. 


#### Notes
- Domain adaptation can be achieved by [Gradient Reversal](https://arxiv.org/abs/1409.7495).
