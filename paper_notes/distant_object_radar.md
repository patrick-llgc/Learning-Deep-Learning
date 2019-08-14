# [Distant Vehicle Detection Using Radar and Vision](https://arxiv.org/abs/1901.10951)

_August 2019_

tl;dr: Fuse radar tracklets with camera image. 

#### Overall impression
Radar as a sensing modality is highly complementary to vision. The paper spatially encoded heterogeneous metadata (radar) as images. This is similar to [meta data fusion for TL2LA](deep_metadata_fusion_tl2la.md).

#### Key ideas
- Automated labeling with long focal length camera to label short focal length camera. 
- Two cameras with different focal length. There is a rotational matrix to connect these two. This is done by manually identified salient points (landmarks).

$$
x_A = K_A R_{AB} K_B^{-1} x_B
$$

- Combine detection from multiple camera to improve quality of detection.
- radar target info is fused with camera through a spatially encoded map for location and velocity.
- Two branches for camera and radar meta data feature map, then fuse. This provides flexibility of reusing RGB features. 
- Precision of the automatically labeled dataset is high with the major issue being low recall. --> **[Missing label is not a big issue for training CNN](https://arxiv.org/pdf/1806.06986.pdf).**
- Range info can be used to boost object detection. --> sensor fusion can do the same!

#### Technical details
- Radar acquisition at 20 Hz. The radar is dual-beam with wide angle (> 90 deg) medium and forward facing narrow beam (< 20 deg). Each has a max of 64 targets. 
- The two bi-focal camera are with a baseline of 3.2 cm, and this leads to ~1 px error in assuming they share the same camera center (thus only translation is needed).
- Camera at 30 Hz. Only pick radar/camera pair that are 10 ms close to each other. --> **Maybe timestamp is enough after all?**
- The camera images look 2x1 aspect ratio because of cropping of hood off images. 
- Small objects: < 20x20 pix, Medium objects: (20 - 60)^2 pix, Large objects: > 60x60 pix.

#### Notes
- Maybe accurate timestamp is sufficient?
- Maybe using different sensor frequency works better than use the same frequency to avoid pathological "phase lock" (exactly interleaved). Oversampling is a special case of different sensor frequency.

