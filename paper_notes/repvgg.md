# [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)

_January 2021_

tl;dr: Train inception-style, inference 3x3 only. Very deployment-friendly.

#### Overall impression
From the authors of [ACNet](acnet.md).

The paper introduces a simple algebraic transformation to re-parameterization technique to transform a multi-branch topology into a stack of 3x3 conv and ReLUs.

> Depthwise conv and channel shuffle increase the memory access cost and lack support of various devices. The amount of FLOPs does not accurately reflect the actual speed.

The structural reparameterization reminds me of the inflated 3D conv [I3D](quo_vadis_i3d.md) initiated from 2d convs. 

This seems to be an architecture very friendly to be deployed on embedded device.

#### Key ideas
- Training has three branches: 3x3, 1x1 and shortcut
- Reparam: 
	- identity branch can be regarded as a degraded 1x1 conv
	- 1x1 can be further regarded as a degraded 3x3 conv
	- all 3 branches can be consolidated into a single 3x3 kernel.
- First layer of every stage is 3x3 conv with stride=2. No maxpooling in RepVGG architecture. 

#### Technical details
- Nvidia cuDNN and Intel MKL (math kernel lib) have accelerations for 3x3 kernels (usually to 4/9 of the original cost) through **Winograd algorithm**.
- ResNet architecture limits the flexibility
	- Limits the tensor shapes due to skip connection
	- Limits channel pruning

- RepVgg is a special type of [ACNet](acnet).
![RepVgg](https://pic3.zhimg.com/80/v2-686b26f8a41b54c10d76d7a90a6d8bbe_1440w.jpg)
![ACNet](https://pic3.zhimg.com/80/v2-c530c6327fbc39319f6c44eca3291e12_1440w.jpg)


#### Notes
- [code on github](https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py)
- [Review of winograd algo](https://www.cnblogs.com/shine-lee/p/10906535.html)

