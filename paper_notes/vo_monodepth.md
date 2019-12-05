# [VO-Monodepth: Enhancing self-supervised monocular depth estimation with traditional visual odometry](https://arxiv.org/abs/1908.03127)

_December 2019_

tl;dr: Use sparse density measurement from VO algorithm to enhance depth estimation.

#### Overall impression
This paper combines the idea of depth estimation with depth completion. 

#### Key ideas
- The paper used a sparsity invariant autoencoder to densify the sparse measurement before concatenating the sparse data with RGB input.
- Inner Loss: between SD (sparse depth) and DD (denser depth after sparse conv)
- Outer loss: between SD and d (dense estimation) on where the SD is defined. 

#### Technical details
- VO pipeline only provides ~0.06% of sparse depth measurement. 
- [Sparcity invariant CNNs](https://arxiv.org/abs/1708.06500) performs weighted average only on valid inputs. This makes the network invariant to input sparsity.

#### Notes
- Best supervised mono depth estimation: DORN
- Scale recovery method is needed for monodepth estimation and any mono VO methods.
- Both ORB-SLAM v1 and v2 supports mono and stereo.