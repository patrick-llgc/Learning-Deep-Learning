# [LiDARsim: Realistic LiDAR Simulation by Leveraging the Real World](https://arxiv.org/abs/2006.09348)

_June 2020_

tl;dr: Generate a map, then place bank of cars on it to create synthetic scenes. 

#### Overall impression
Lidar sim is similar to [surfel GAN](surfel_gan.md) in generating synthetic dataset with real data collection. 

- [surfel GAN](surfel_gan.md) generates a photorealistic model
- [lidar sim](lidar_sim.md) focuses on lidar data simulation, which is somewhat easier. 

It can allow closed-loop evaluation of the whole AD stack.

#### Key ideas
- Simulate ray drop patterns with U-Net structure
- Minimum sim2real domain gap.
 
#### Technical details
- Chart showing the diversity of cars on the road
![](https://cdn-images-1.medium.com/max/1280/1*g4byc9o0saZQrfbrW8BkQg.png)
![](https://cdn-images-1.medium.com/max/1280/1*8wTEcAy97RWFIB4S_xeY6g.png)

#### Notes
- Questions and notes on how to improve/revise the current work  

