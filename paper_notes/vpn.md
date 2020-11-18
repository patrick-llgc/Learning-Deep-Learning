# [VPN: Cross-view Semantic Segmentation for Sensing Surroundings](https://arxiv.org/abs/1906.03560)

_September 2020_

tl;dr: Generate cross-view segmentation with view parsing network.

#### Overall impression
The paper could have been written more clearly. Terminology is not well defined and the figure is not clear.

The idea of using semantic segmentation to bridge the sim2real gap is explored in many BEV semantic segmentation tasks such as [BEV-Seg](bev_seg.md), [CAM2BEV](cam2bev.md), [VPN](vpn.md).


#### Key ideas
- **View transformation**: MLP
	- View Relation Module (VRM) to model the transformation from perspective to BEV. The view transformation is learned with a MLP with flattened image input HW x C.
- Using synthetic data to train. Use adversarial loss for domain adaptation. **Semantic mask** as intermediate representation without texture gap.
- Each perspective (first-view) sensor data is transformed with its own View Transformer Module. The Feature map is then aggregated into one BEV feature map. The BEV feature map is then decoded into a BEV semantic map.

#### Technical details


#### Notes
- The BEV feature map has the same shape as the input feature map. --> Why is this necessary?
- How was the fusion of feature map done?
- [Review from 1st author on Zhihu](https://mp.weixin.qq.com/s/8jltlOnAxK1EqxYCsJHErA)