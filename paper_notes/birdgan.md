# [BirdGAN: Learning 2D to 3D Lifting for Object Detection in 3D for Autonomous Vehicles](https://arxiv.org/abs/1904.08494)

_October 2019_

tl;dr: Learn to map 2D perspective image to BEV with GAN.

#### Overall impression
The performance of BirdGAN on 3D object detection has the SOTA. The AP_3D @ IoU=0.7 is ~60 for easy and ~40 for hard. This is much better than the ~10 for [ForeSeE](foresee_mono3dod.md)

One major drawback is the limited forward distance BirdGAN can handle. In the clipping case, the frontal depth is only about 10 to 15 meters. 

Personally I feel GAN related architecture not reliable for production. The closest to production research so far is still [pseudo-lidar++](pseudo_lidar++.md).

#### Key ideas
- Train a GAN to translate 2D perspective image to BEV. 
- Use the generated BEV to perform sensor fusion in AVOD and MV3D. 
- Clipping further away points in lidar helps training and generates better performance --> while this also severely limited the application of the idea.

#### Technical details
- Summary of technical details

#### Notes
- Maybe the 3D AP is not what matters most in autonomous driving. Predicting closeby objects better at the cost of distant objects is not optimal for autonomous driving.