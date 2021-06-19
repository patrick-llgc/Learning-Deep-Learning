# [FIERY: Future Instance Prediction in Bird's-Eye View from Surround Monocular Cameras](https://arxiv.org/abs/2104.10490)

_June 2021_

tl;dr: Prediction in BEV from monocular cameras. 

#### Overall impression
This paper is heavily inspired by [Lift Splat Shoot](lift_splat_shoot.md) in lifting multicamera features to 3D and then splat onto BEV view. However they are different too. 

- [Lift Splat Shoot](lift_splat_shoot.md) focuses on motion planning of ego car in the "shoot" part, while [FIERY](fiery.md) focuses on behavior prediction of other traffic participants.
- [FIERY](fiery.md) improves the semantic segmentation of [Lift Splat Shoot](lift_splat_shoot.md) to instance segmentation.
- [FIERY](fiery.md) also introduced the temporal component and leverages past frames.



#### Key ideas
- Summaries of the key ideas

#### Technical details
- The BEV backbone of combining multiple cameras has the functionality of sensor fusion. Instead of [Lift Splat Shoot](lift_splat_shoot.md) that does wholistic motion planning directly, [FIERY](fiery.md) actually does the prediction first, and the authors mentioned that they will work on the planning part later. It is a bit like [MP3](https://arxiv.org/abs/2101.06806) by Uber ATG.

#### Notes
- Code available at [github](https://github.com/wayveai/fiery).
