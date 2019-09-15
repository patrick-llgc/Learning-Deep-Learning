# [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)

_September 2019_

tl;dr: Seminal paper from MSRA that improves upon faster R-CNN. 

#### Overall impression
Faster RCNN computation increases as ROI number grows, as each ROI has a fully connected layer. R-FCN improves the computation efficiency by moving the FCN to before ROI pooling by generating position sensitive score maps (feat maps). **Each PS score map is responsible to fire at a particular region (top-left corner) of a particular class.**

However R-FCN cannot leverage FPN directly as the number of channels are too large for large dataset such as COCO. This is improved in [Light-head RCNN](lighthead_rcnn.md) to reduce the number of score maps from #class x p x p to 10. 

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

