# [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)

_June 2019_

tl;dr: FCOS leverages all pixels in the bbox for regression, instead of just the center, to alleviate the imbalance problem. Each pixel in the feature map is responsible for regress four distances to the four edges of GT bbox.

#### Overall impression
This paper is littered with hidden gems! Lots of tricks and insights on training neural nets or architecture design. The paper assumes bbox annotation. If mask is also available, then we could use only the pixels in the mask to perform regression.

The idea is similar to [CenterNet](centernet.md). CenterNet uses only the points near the center and regresses the height and width, whereas FCOS uses all the points in the bbox and regresses all distances to four edges. **FCOS does not predict centers of bbox using keypoints heatmap but rather uses anchor free methods.**

[FCOS](fcos.md) regressed distances to four edges, while [CenterNet](centernet.md) only regresses width and height. The [FCOS](fcos.md) formulation is more general as it can handle amodal bbox cases (the object center may not be the center of bbox).

I personally feel this paper is better than centernet in the sense that it does not need too much bells and whistles to achieve the same performance. 

It is extended to [PolarMask](polarmask.md) for one-stage instance segmentation.

The paper inspired [ATSS](atss.md) which explained why FCOS can achieve better performance than RetinaNet.

#### Key ideas
- Proposal free and anchor free, significantly reduces the number of design parameters.
- Use multilevel FPN prediction to alleviate overlapping bounding box issue (standard practice with FPN).
- Use IoU loss for bbox regression loss, focal loss for classification loss. They also used the classification branch to do perform centerness regression.
- Centerness score = $\sqrt{\frac{\min(l, r) \min(t, b)}{\max(l, r) \max(t, b)}}$
- Assign regression target to different levels. For central points, use high-res feature map, and for corner points, use low-res feature map. 
- Regress a centerness score in the classifier branch to filter out FP bboxes far away from center. This bridges the gap between previous SOTA and FCOS.
- The performance of FCOS is better than RetinaNet at stricter thresholds.

#### Technical details
- Even for RetinaNet, using GN instead of BN helps a lot.
- For always positive regression target, reparameterize as exp(x) and regress x from neural network. Use exp(x) in the loss function.
- Use cross entropy loss to regress loss for prediction and target in [0, 1].
- Share the same head across regions but to align the regression target range, regress exp(sx) instead of x. In the original FPN and RetinaNet paper, this is not an issue as the regression target are all relative to w and h. 
- Best possible recall (**BPR**) may be bad for anchor-based methods due to large strides in feature maps, FCOS has a higher BPR. **A gt is considered recalled if the box is assihgned to at least one sample (a location in FCOS or anchor box in anchor-based methods.**
- Centerness is only used in the postprocessing. Therefore it is possible to train a separate net for better prediction and improve performance. (GT centerness leads to huge improvement in AP, therefore leaving enough headroom to explore.)

#### Notes
- Q: maybe use the centerness score to modulate the regression loss? Now the paper just use centerness loss in the deployment to downweight the bbox far away from object center. --> maybe this become more like centernet?
- If regression target is in [0, 1], we can use cross entropy.
- Fig. 7 is very interesting. It means if the predicted (classification/confidence) score is 0.6, the IoU is larger than 0.6. 
- The centerness score is regressed in the classifier branch, perhaps mainly due to the loss function (CE) used. --> Regressing it in the other branch or even a separate branch may lead to a better performance. 
- How fast is the network? --> 41 AP @ 70 ms as per [github repo](https://github.com/tianzhi0549/FCOS). On par with CenterNet. 
- The idea of using cls score to predict IOU is very good. We should try it in error analysis.
