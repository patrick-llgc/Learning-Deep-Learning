# [High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection](https://arxiv.org/pdf/1904.02948.pdf) (center and scale prediction)

_April 2019_

tl;dr: an anchor free method to directly predict the center and scale of pedestrian (single class) bounding boxes. It was heavily influenced by [CornerNet](cornernet.md) but reformulate the object detection task to get rid of the data association problem. 

#### Overall impression
The architecture is surprisingly simple. Previous methods require tedious config in windows or anchors. The method is less prone to occlusion of bbox as it predicts the center of amodal bbox directly. Also it only focuses on the binary classification problem. How to extend this to general object detection remains to be explored.

#### Key ideas
- Two classical ways to do object detection: sliding window, and anchor-based prediction in deep learning era.
- The previous methods couples where and how subproblems into a single one through the overall judgement of various scales of windows or anchors. 
- Anchor-free object detection including CornerNet and TLL recently reached the SOTA. However both methods regresses to the corner instead of center point. 
- The authors showed that using center is better than corner. It may be due to that regressing corners requires a large field of view (or maybe it does not require to see the full image to set the corner?). 
- The aspect ratio is fixed at 0.41 for pedestrian. Only the height is annotated, and the width is generated automatically.
 
#### Technical details
- Deconv after stage 5 to keep it 1/16 of original image.
- *L-2 normalization* to rescale feature maps before concatenation. ([source](https://arxiv.org/pdf/1506.04579.pdf)) Alternatively, use a conv layer before concatenation (for scaling), such as is done in FPN. 
- Log-average Miss Rate over False Positive Per Image ranging [0.01, 1] $MR^{-2}$ KPI for pedestrian detection, at IoU=0.5 and 0.75. This is used in Caltech and Citypersons datasets.
- Focal loss is used to tackle the extreme class imbalance problem.
- A (inverse-) gaussian mask surrounding the center of the object is used to weigh the contributions to loss of each pixels surrounding the positive pixel (center pixel), similar to [CornerNet](cornernet.md). For negative points very close to ground truth, the loss is weighted as 1-gaussian.
- Citypersons KPI divides object to reasonable, heavy, partial, bare, small, medium, large. This may be useful for vehicle detection as well.


#### Notes
- The general-purpose object detection and target-specific object detection have different requirements and leads to different SOTA architecture. The general purpose object detection has to accommodate wide range of sizes and aspect ratios, but special purpose ones require fast inference time (pedestrian detection, etc) but appearances of objects may be limited. 
- RepLoss and OR-CNN to tackle occluded pedestrian detection in crowded scene.
- The paper did not offer a compelling explanation why center is better than corners for object detection. However in order to predict the center of an object it must see the whole object and thus is more robust to occlusion. 
