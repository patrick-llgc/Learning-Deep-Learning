# [Amodal Completion and Size Constancy in Natural Scenes](https://arxiv.org/abs/1509.08147)

_January 2020_

tl;dr: Infer the horizon and veridical size of objects in 2D images, with amodal 2D object detectors.

#### Overall impression
The paper explores the capability of using NN to predict the whole physical extent of object in 2D bbox, even though the object may be occluded or truncated. 

> "Almost nothing is visible in its entirety, yet almost everything is perceived as a whole and complete."

The study did not start with amodal object detector, but rather using modal object detector and the image context in the modal bbox to infer the whole extent of amodal bbox, by explicitly modeling occlusion patterns along with detections.

> The hypothesis of amodal completion is that the amodal prediction task can be reliably addressed given just the image crrrespodonhg top the visible object region (seeing the left of a car is sufficient to unambiguously infer the fully extent without significantly leveraging context).

Three important cues for depth perception: familiar size and relative size and perspective position. 

- Familiar size is what we explored by using the apparent size and metric size of object to infer absolute depth. 
- Relative size is further objects appearing smaller (近大远小). 
- Perspective position is that further objects appearing higher on the image (近低远高).

There is a strong assumption that all objects detected are on the ground, which may not be true, even in the sample images shown.

#### Key ideas
- Three steps to estimate veridical/truthful size of objects
	- Amodal completion. Regress left/right edge, bottom of y and height. --> **Bottom y is more important than top y.**
	- Geometric reasoning with size constraints. Learn probabilistic object size distribution, with completed amodal bbox to infer relative real object size
	- Predict focal length for metric size recovery
- The GT is recovered with PASCAL 3D.
- Pairwise height ratio (Ground contact point, similar to IPM)
$$\frac{H_i}{H_j} = \frac{h_i}{h_j} \frac{y_j^b - y_h}{y_i^b - y_h}$$
	- Solve for logH to linearize constraints
	- Gaussian mixture of logH --> This aligns with psycophysics studies that our mental representation of object sizes
	- recovers horizon as well
- Focal length prediction
	- Classification
	- Hint: scenes taken with longer focal length is less cluttered

#### Technical details
- Training amodal completion is carried out with jittering instances of GT box, to enable generalization from noisy settings such as detection boxes and serve as data augmentation.
- Class agnostic completion is as good as class specific one

#### Notes
- For autonomous driving, there are some takeaways
	- Predicting occluded bbox should be OK for neural networks, but how to get the GT is tricky. Asking labers to label amodal box is a challenging task. Maybe we need to do more augmentation with random input block dropout (cf. CutOut)?
	- Solving pairwise constraints may be needed to recover robust distance for monocular perception (cf [ApolloCar3D](apollocar3d.md))
	- logH to linearize the constraints
	- Recovery of horizon is approximate. If there is only one object, there is no way to recover horizon line. This is somewhat related to recover horizon line with MOD detections.


