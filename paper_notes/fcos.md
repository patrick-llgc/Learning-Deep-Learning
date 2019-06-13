# [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)

_June 2019_

tl;dr: Use all pixels in the bbox for regression, instead of just the center. Each pixel in the feature map is responsible for regress four distances to the four edges of GT bbox.

#### Overall impression
This paper is littered with hidden gem!

This paper is very well written with a lot of insights on architecture design.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Even for RetinaNet, using GN instead of BN helps a lot
- s

#### Notes
- Q: maybe use the centerness score to modulate the regression loss? Now the paper just use centerness loss in the deployment to downweight the bbox far away from object center. --> maybe this become more like centernet?
- If regression target is in [0, 1], we can use cross entropy.
- Fig. 7 is very interesting. It means if the predicted (classification/confidence) score is 0.6, the IoU is larger than 0.6. 
