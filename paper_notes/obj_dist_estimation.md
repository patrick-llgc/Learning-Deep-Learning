# [Learning Object-specific Distance from a Monocular Image](https://arxiv.org/abs/1909.04182)

_October 2019_

tl;dr: Train a neural network to learn distance from patches.

#### Overall impression
The use of CNN to directly predict distance for given objects in the image. Extract features from patches should be effective in predicting distance. 

The paper also demonstrated that the vehicle category is key to the good performance, which makes perfect sense as the size of the same type of vehicle is small. 

Dense depth estimation is too costly in both memory footprint and processing time for autonomous driving and object level distance should be good enough.

Refer to [DistNet](distnet.md) for a much simpler way to estimate depth.

#### Key ideas
- RoIAlign features according to bbox and pass through fc layer to predict distance. 
- Network is trained with auxiliary vehicle type classifier branch.
- Absolute Relative error is about 0.15. --> 10% error may be too much to ask for?

#### Technical details
- The use of softplus to make sure the target is positive. (The derivative of softplus is sigmoid)


#### Notes
- Ablation study: how about noisy bbox? The bbox seems to be the GT. 
- The projection loss of keypoint is really doubtful, as the keypoint is picked as the 10th percentile according to distance in the point cloud and there is no visual consistency to it. I think the model is overfitted on the validation set. 
- No specific HW reported for inference. 
- How about feeding category and bbox size directly into a much shallower regressor?
- How does this compare with monocular 3D method? The 3D method seems to have better performance.