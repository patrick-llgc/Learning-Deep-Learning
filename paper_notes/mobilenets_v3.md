# [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)

_May 2019_

tl;dr: Combination of automatic search (NAS and NetAdapt) with novel architecture advances (SENet, swish activation function, hard sigmoid) to search for MobileNetv3.

#### Overall impression
Improved upon [MobileNetsv2](mobilenets_v2.md) (with inverted residuals and linear bottleneck) and [MNasNet](mnasnet.md) (NAS-optimized MobileNetsv2). The paper itself does not dwell too much on NAS, but instead reported the searched result, a deterministic model, similar to MNasNet. In particular, MobileNetV3-large uses MnasNet as baseline and uses net-adapt to finetune.

The idea of NetAdapt seems to be practical. It is complementary to NAS and finetunes the number of layers, after NAS finalizes the number of blocks.

The most useful takeaway is the MobileNetV3-Large and MobileNetV3-Small backbones.  See PyTorch Reimplementation on [github](https://github.com/xiaolai-sqlai/mobilenetv3).

MobileNet series talks about how to be fast yet accurate. [EfficientNet](efficientnet.md) talks about how to scale up MobieNet to achieve SOTA accuracy yet efficient. 

#### Key ideas
- Edge computation is key enabler for personal privacy.
- SqueezeNet is designed for reduction in num of parameters and model size. Recent efforts focuses more on reduction in explicit FLOPS and inference time.
- Two steps:
	- MNasNet uses platform-aware NAS for block-wise search, and uses a cost function $ACC(model) \times [LAT(model)/TAR]^w$, with w=-0.07~-0.15. -0.07 was obtained by observing that empirically model accuracy improves by 5% when doubling latency.
	- [NetAdapt](https://arxiv.org/pdf/1804.03230.pdf) finetunes the number of filters in each layer.
- New activation functions
	- ReLU6 caps ReLU at 6, and more quantization friendly.
	- Hard sigmoid = ReLu6(x+3)/6
	- Swish (also Nas'ed) = $x \sigma(x)$. **Note that swish is not monotonic! Swish usually works better in deeper layers.**
	- h-swish = x ReLu6(x+3)/6, used in this work
- The paper also proposed Lite R-ASPP segmentation head.
- The paper improved performance on imagenet significantly (x2 reduction of inference resources). But on object detection, the improvement is less. It seems to be overfit on imagenet.
- On segmentation task on cityscape, reducing channels in the last block does not hurt performance. This makes sense as cityscape has very few object classes. No need to have as many channels as optimal models for COCO or imagenet.

#### Technical details
- Platform aware test is done with TFLite Benchmark Tool.
- 1 MAdd is generally 2 FLOPS. MobileNetsv1 is 300M MAdds, 600M FLOPs.

#### Notes
- Activation functions do not have to be monotonic, such as swish. 