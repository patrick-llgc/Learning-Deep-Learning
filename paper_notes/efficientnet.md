# [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

_June 2019_

tl;dr: Scaling network up jointly by resolution, depth and width is a wiser way to spend inference budget. 

#### Overall impression
The paper proposed a simple yet principled method to scale up networks. The main difference from previous work is the exploration of image resolution in addition to network architecture (width and depth). 

**The EfficientNet-B0 is similar to MnasNet**. Scaling this network up pushes the Pareto frontier for imageNet significantly, achieving similar accuracy with x10 reduction in FLOPs and parameters. In other words, beefed-up MobileNet beats SOTA such as ResNet and ResNeXt. EfficientNets studies how to spend more resource wisely. The depth, width and resolution scaling factors are usually larger than 1. 

On the other hand, the mobilenets papers ([v1](mobilenets_v1.md), [v2](mobilenets_v2.md) and [v3](mobilenets_v3.md)) goes the other way round. They start with an efficient network and scales it down further. The channel and resolution scaling factors are usually smaller than 1. Note that **MobileNetv3-Large optimizes based on MnasNet**. Therefore EfficientNet-B* is really all about how to scale up MobileNet, and tells us that a beefed-up MobileNet is better than ResNet. In the original [MobileNetsv1](mobilenets_v1.md)

This paper inspired follow-up work [EfficientDet](efficientdet.md), also by Quoc Le's team.

#### Key ideas
- The balance of width/depth/resolution can be achieved by simply scaling each of them with constant ratio.
	- Deeper network captures richer and more complex features
	- Wider networks tend to be able to capture more fine grained features and re easier to train
	- Input image with higher resolution requires bigger network to process but also leads to improvement in accuracy.
- The effectiveness of model scaling heavily depends upon the baseline network.
- Observation 1: Scaling up any of width/depth/resolution improves accuracy, but the gain diminishes for bigger networks
- Observation 2: it is critical to balance all dimensions of network width, depth and resolution during scaling.
- Compound scaling method:
	- depth: $d=\alpha^{\phi}$
	- width: $w=\beta^{\phi}$
	- resolution: $r=\gamma^{\phi}$
	- s.t. $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$, $\alpha, \beta, \gamma \ge 1$
	- The three constants can be determined by a small grid search

#### Technical details
- Blindly scaling up only one dimension hits plateau very quickly
![](https://www.groundai.com/media/arxiv_projects/551726/x8.png)

#### Notes
- The scaling ratios of EfficientNet-B0 to B7 can be found in the [github repo](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py).

```python
params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5),
  }
```
