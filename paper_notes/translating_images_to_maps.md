# [Translating Images into Maps](https://arxiv.org/abs/2110.00966)

_December 2021_

tl;dr: Axial transformers to lift images to BEV.

#### Overall impression
The paper assumes a 1-1 correspondence between a vertical scanline in the image, and rays passing through the camera location in an overhead map. This relationship holds true regardless of the depth of the pixels to be lifted to 3D. 

This paper is written with unnecessarily cumbersome mathematical notation, and many concepts can be explained in plain language with transformers terminology.

#### Key ideas
- Ablation studies
	- Looking both up and down the same column of image is superior to looking only one way (constructed with MAIL -- monotonic attention with infinite look-back).
	- Long range horizontal context does not benefit the model.

#### Technical details
- The optional dynamic module in BEV space uses axial-attention across the temporal dimension (stack of spatial features along temporal dimension). This seems to be less useful without spatial alignment as seen in [FIERY](fiery.md).

#### Notes
- [Code on Github](https://github.com/avishkarsaha/translating-images-into-maps) to be released.
