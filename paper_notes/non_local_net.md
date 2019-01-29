# [Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)

_Jan 2019_

tl;dr: Use a non-local module to capture long range dependencies (spatial, time or spatialtime).

#### Overall impression
The design of non-local module is modular, which does not change the input shape. Therefore it can be plugged into existing network design easily. This is similar to the deisgn of SE-net. The authors show that self-attention proposed in Google's transformer paper is a special case (embedded Gaussian) of non-local networks.

#### Key ideas
- Both CNN and RNN blocks are local operations, and model long range dependencies progressively through repeated local operations.
- The **formulation** of non-local operation is as follows, where the output feature z has the same shape as the input feature x to ensure a modular design. $\hat{x}_j$ can be the original or subsampled $x_j$ or some other features (such as in [long term feature bank](long_term_feat_bank.md))

$$
y_i = \frac{1}{C(x)}\sum_{\forall{j}} f(x_i, \hat{x}_j) g(\hat{x}_j) \\
z_i = \text{BatchNorm}(W_z y_i) + x_i
$$

- The non-local operation is different from a fully connected (fc) layer. Non-local is more flexible in that the output size matches the input size and can be inserted anywhere inside a network and keep spatialtime information.
- The embedded Gaussian version of non-local is the self-attention module.
- Used alone, non-local + 2D is better than 3D counterparts. 
- The non-local module can also improve static image detection tasks, such as Mask RCNN on COCO. 

#### Technical details
- The **instantiation** of non-local net can take on many forms, but the most common/generic form is as follows. $F(x)$ can be a gaussian or identify function, leading to embedded gaussian or dot product. The gaussian is the dot product plus a softmax.

$$
g(x_j) = W_g x_j \\
f(x_i, x_j) = F(\theta(x_i)^T \phi(x_j)) \\
$$

- There is a batch-norm layer before residual sum with original x, initiated with zero weight to ensure smooth fine-tuning process.
- Input feature $x$ is first linearly embedded to reduce the number of channels by half. Pooled features $x_j$ are used to augment input features $x_i$. 
- In Mask RCNN, non-local block can be added to backbone (e.g., right before last resblock res4) or in the head (e.g., after every 2 layers for keypoint detection) or both.

#### Notes
- Use non-local module in Mask RCNN to boost performance.
- The paper introduced a variant of convnet called C2D, which uses 2D weight directly on 3D data. Only the parameter-less pooling operation is used to capture dependencies along the 3rd dimension.
