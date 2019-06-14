# [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)

_June 2019_

tl;dr: FCOS leverages all pixels in the bbox for regression, instead of just the center, to alleviate the imbalance problem. Each pixel in the feature map is responsible for regress four distances to the four edges of GT bbox.

#### Overall impression
This paper is littered with hidden gems! Lots of tricks and insights on training neural nets or architecture design. The paper assumes bbox annotation. If mask is also available, then we could use only the pixels in the mask to perform regression.

The idea is similar to [CenterNet](centernet_ut.md). CenterNet uses only the points near the center and regresses the height and width, whereas FCOS uses all the points in the bbox and regresses all distances to four edges.

#### Key ideas
- Proposal free and anchor free, significantly reduces the number of design parameters.
- Use multilevel FPN prediction to alleviate overlapping bounding box issue (standard practice with FPN).
- Use IoU loss for bbox regression loss, focal loss for classification loss. They also used the classification branch to do perform centerness regression.
- Best possible recall

#### Technical details
- Even for RetinaNet, using GN instead of BN helps a lot
- For always positive regression target, reparameterize as exp(x) and regress x from neural network. Use exp(x) in the loss function.
- Use cross entropy loss to regress loss for prediction and target in [0, 1].

#### Notes
- Q: maybe use the centerness score to modulate the regression loss? Now the paper just use centerness loss in the deployment to downweight the bbox far away from object center. --> maybe this become more like centernet?
- If regression target is in [0, 1], we can use cross entropy.
- Fig. 7 is very interesting. It means if the predicted (classification/confidence) score is 0.6, the IoU is larger than 0.6. 
