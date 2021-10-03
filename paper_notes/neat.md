# [NEAT: Neural Attention Fields for End-to-End Autonomous Driving](https://arxiv.org/abs/2109.04456)

_October 2021_

tl;dr: transformers to learn an interpretable BEV representation for end-to-end autonomous driving.

#### Overall impression
The goal of the paper is interpretable, high-performance, end-to-end autonomous driving. Yet the way to generate the interpretable intermediate representation is quite interesting.

Both [NEAT](neat.md) and [PYVA](pyva.md) also uses the idea of transformers in the task of lifting image to BEV, however upon closer look, the transformers are not used specifically for view transformation. 

- [NEAT](neat.md) uses transformers to enhance the features in image feature space, before using MLP-based iterative attention to lift image features into BEV space. 
- [PYVA](pyva.md) uses MLP to lift image features to BEV space first and then uses a transformer to enhance the features in BEV. 
- Both of them are quite different from Tesla's approach.

The paper has many esoteric details regarding planning and control, and I am not sure if I fully understands those parts, and the discussion of those parts are ignored here.

#### Key ideas
- Encoder
	- $R^{S \times T}$

#### Technical details
- The C-channel features $c_{i-1} \in R^C$ are initiated as the average of $C \in R^{(S*T*P)\times C}$. This should be largely equivalent to the context summary in Tesla's AI day presentation.
- The MLP in the **neural attention field** module should have taken p (query, 5) and a fully-fledged image feature c ($STP \times C$). Then it should have $(STP)$ dimension attention map, then the output is a softmax-scaled dot product between c and the attention map. In Tesla's AI day presentation, the coordConv is replaced with a positional embedding, as is in trasnformers.
	- In this paper, the authors argue that this approach is too inefficient due to the high dimension of c ($STP \times C$). Therefore it average pools c into c_0 ($C$) and feed into the MLP. As the input of the MLP does not have any spatial information regarding the original c (spatial content already reduced to 1x1 by global averaging), it has to use N iterations to refine. In this sense, the NEAT is a simplified version of Transformers. --> **Not super sure the rationale behind the N iterations of this approach.**
- The MLP in decoder takes in 5 + C (query + C-channel features) as input, and iterates this on all possible query locations (2) in the BEV space (More precisely, location (x,y) + time + target waypoint location (x', y'), thus 5 dimensions. But we will ignore the latter 3 here for simplicity). This is equivalent to have a raster of BEV space with [CoordConv](coord_conv.md) and tile the globally (average) pooled in BEV space, and perform a 1x1 convolution on top of it. This 1x1 convolution keeps the spatial dimension to the same (BEV h x w) and channel is changed from 5+C to K+2 (K is semantic classes, 2 is offset prediction). 
	- This decoder part is not detailed in Tesla's AI day presentation. And the MLP should be equivalently replaced by a 1x1 convolution. In practice, 3x3 convolutions may be even better in the decoder of BEV features. 

![](https://cdn-images-1.medium.com/max/2400/1*4bnC51_5JIzDom2llXjWGg.png)
![](https://cdn-images-1.medium.com/max/1600/1*QLaCIWYMFrlSy2Mm8Wv4xQ.png)

#### Notes
- [Code on github](https://github.com/autonomousvision/neat)

