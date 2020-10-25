# [ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)

_October 2020_

tl;dr: Break images into images patches as visual tokens to leverage the scalability of transformers.

#### Overall impression
This paper, together with earlier efforts from FAIR [DETR](detr.md) ushers in an new era of the application of [transformers](transformer.md) in CV. 

Transformers lack some inductive biases inherent to CNNs, such as translation equivariance and locality, and thus do not generalize well when trained on insufficient amounts of data. However when trained on large amount of data, **large scale training trumps inductive bias**. 

However, the splitting of images into patches itself seems to be a kind of inductive bias to me.

#### Key ideas
- It is not scalable to apply transformers directly in pixel space, as the attention matrix scales quadratically with number of pixels, and thus quatically (4th power) with input image (lateral) size. Previous efforts also reduces images resolution and color space before applying transformers. 
- Split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer.
- **Self attention allows ViT to integrate info across the entire image even in the lowest layers**. The attention distance is more like receptive field in CNN. CNN has to be deep to scale up the receptive field, but transformers do not need to be super deep (32 layers is already the huge setting in ViT).
- Scalability
	- Vision transformers (with similar size to ResNet) **overfit** to smaller datasets more than ResNet. This also shows that convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns is sufficient, even beneficial. The performance of the model does not seem to saturate yet. This in a sense is similar to the scenario of ML vs DL. 
	- Transformer is in a sense a dynamic MLP, where the weights (attention) are generated on the fly.

#### Technical details
- 2D embedding works roughly the same as 1D embedding. Even 1D embedding can learn the 2D image topology row and col wise correlation as visualized in Fig. 7.
- [class] token in Bert. --> Why do we need this?
- Note that the transformer encoder can be stacked by layers. Base model has 12 layers, and Huge ViT model has 32 layers. The input are input into the encoder at the same time, not fed autoregressively, like in RNN.
- Generalize into higher resolution by keeping the patch resolution the same. The positional embedding are 2D-interpolated (this is another source of inductive bias).
- Using smaller resolution patch and longer sequence seems to improve the model performance.

#### Notes
- [Yannic Kilcher's explanation on youtube](https://www.youtube.com/watch?v=TrdevFK_am4)
- [Review on synced](https://mp.weixin.qq.com/s/WC4LiTz7fIr2myl4CEjShw)
- [Review on medium](https://medium.com/swlh/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-brief-review-of-the-8770a636c6a8)
- The paper obviously comes from Google, as Yannic pointed out in his video.

