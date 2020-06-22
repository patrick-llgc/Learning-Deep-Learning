# [SurfelGAN: Synthesizing Realistic Sensor Data for Autonomous Driving](https://arxiv.org/abs/2005.03844)

_June 2020_

tl;dr: Drive once, build, and traverse multiple times to generate new data. 

#### Overall impression
Use GAN to bridge the gap of surfel baseline, which is usually with gaps and edges. 

**Surfel** is "surface element," analogous to a "voxel" (volume element) or a "pixel" (picture element).

- [surfel GAN](surfel_gan.md) generates a photorealistic model
- [lidar sim](lidar_sim.md) focuses on lidar data simulation, which is somewhat easier. 

It can allow closed-loop evaluation of the whole AD stack.

#### Key ideas
- Build environment from a single run through a scene of interests. 
- Simulate other transversals through the scene for virtual replay. 
- Use a GAN model to close the domain gap between synthetic data and real data


#### Technical details
- Uses lidar data

#### Notes
- Questions and notes on how to improve/revise the current work  

