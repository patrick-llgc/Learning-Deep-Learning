# [LSTR: End-to-end Lane Shape Prediction with Transformers](https://arxiv.org/abs/2011.04233)

_November 2020_

tl;dr: Use transformer to directly predict polynomial coefficients of lanes. 

#### Overall impression
The paper uses [transformers](transformer.md) for lane line detections. It is inspired by [DETR](detr.md), which introduced the transformers to object detection.

The paper has a great session talking about the polynomial lane shape model with very detailed derivation. 

The formulation of lane lines as parallel polynomials is a bit limiting as it cannot handle more complex topologies such as splits, merges and lanes perpendicular to the ego lane. However the idea is still applicable if we allow a more flexible representations of lane lines, as long as there is still the concept of individual countable lane line instances (number of query sequence).

The paper recycles a lot of the details from [DETR](detr.md) but describes them differently. It is recommended that these two papers should be read together. 

#### Key ideas
- **Polynomial lane shape model**
	- Cubid curve for a single lane on flat ground (X, Z)
	$$X = kZ^3 + mZ^2 + nZ + b$$
	- Projected to (u, v). The primed parameters are composites of parameters and camera intrinsics and extrinsics.
	$$u = k' / v^2 + m' / v + n' + b' \times v$$
	- For tilted road (with pitch)
	$$u = k'' / (v - f'')^2 + m'' / (v - f'') + n' + b'' \times v - b'''$$
	- Vertical starting and ending offsets $\alpha, \beta$ in images. 
	- If we assume a global consistent shape for all lanes, then $k'', f'', m'', n'$ will be shared for all lanes, and $b'', b'''$ will not be shared. 
	- Therefore, the output of t-th lane is 
	$$g_t = ((k'', f'', m'', n'), (b_t'', b_t''', \alpha_t, \beta_t))$$. 
	- Each lane line is only different in bias terms and starting and ending positions. 
- Loss
	- Hungarian bipartite matching
		- Largely follow [DETR](detr.md). 
		- N=7 lane lines at most. In case there are fewer than N lane lines in GT, pad with non-lanes with cls score c=0. 
	- Location loss
		- L1, sample lanes at fixed intervals. Starting and ending positions are also predicted. 
- Architecture
	- ResNet-18 backbone
	- Input: HxWxC feature map is reshaped into HW sequence of length C. Concatenated with positional embedding (not necessarily the same size as input as the paper claims).
	- Encoder: two layers. QKV are all front the input sequence with length HW (each has a learnable matrix)
	- Decoder: two layers. Initial input to decoder $S_q$ are all zeros. Positional embeddings $E_{LL}$ are learned specialized workers attending to different types of lane lines. --> This is not described clearly in the original [DETR](detr.md) paper. 
- FNN for predicting curve parameters
	- Nx2 for predicting each lane is lane or background
	- Nx4 to predict 4 specific parameters
	- Nx4 followed by average to regress shared 4 parameters 

#### Technical details
- Too many layers of encoder and decoder are causing overfitting and degraded generalization ability.

#### Notes
- [Github repo](https://github.com/liuruijin17/LSTR)

