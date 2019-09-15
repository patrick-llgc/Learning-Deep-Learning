# [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)

_September 2019_

tl;dr: Seminal paper from MSRA that improves upon faster R-CNN. 

#### Overall impression
Faster RCNN computation increases as ROI number grows, as each ROI has a fully connected layer. R-FCN improves the computation efficiency by moving the FCN to before ROI pooling by generating position sensitive score maps (feat maps). **Each PS score map is responsible to fire at a particular region (top-left corner) of a particular class.**

Note that usually R-FCN has slightly lower performance, especially compared to FPN-powered Faster RCNN.

R-FCN cannot leverage FPN directly as the number of channels are too large for large dataset such as COCO. This is improved in [Light-head RCNN](lighthead_rcnn.md) to reduce the number of score maps from #class x p x p to 10. Instead, the simple voting mechanism is replaced by a fully connected layer.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Summary of technical details

#### Notes
- [This medium blog post from Jonathan Hui](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99) explains the intuition very well.