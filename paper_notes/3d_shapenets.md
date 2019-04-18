# [3D ShapeNets: A Deep Representation for Volumetric Shapes](http://3dshapenets.cs.princeton.edu/paper.pdf)

_Mar 2019_

tl;dr: A convolutional deep belief network (CDBN) is trained to perform recognition and retrieval of 3D voxel grid. It can also hallucinate the missing parts of depth maps.

#### Overall impression
The paper builds upon deep belief network popular at that time and uses a new way to train 3D network. It also achieves several other tasks, such as 2.5D joint classification and completion, and next best-view prediction. However it is surpassed by many other methods later such as [MVCNN](mvcnn.md) and the famed [PointNet](pointnet.md).

This paper is from CVPR 2015, only 4 years old as of the time of writing but I am already having difficulty going through this paper as smoothly as other more recent papers, due to the archaic terminology and method.

#### Key ideas
- ShapeNet represent a geometric 3D shape as a probabilistic distribution of binary variables on a 3D voxel grid.
- The 2.5D depth map is first converted to a volumetric representations, consisting of pixels categorized as free space, surface or occluded. $x = (x_o, x_u)$, *observed* space and *unknown* space.
- [Gibbs sampling](https://www.youtube.com/watch?v=a_08GKWHFWo) is used to estimate the posterior distribution $p(y|x_o)$. 
- Next-Best-View prediction predicts which view will leads to the largest entropy drop. This will first needs the network to hallucinate enough 3D shapes and then calculate the entropy drop by shining light from different views. 

#### Technical details
- The 3D convnet is trained in a layer-wise fashion followed by a fine-tune procedure. This is very similar to RBM.
- A joint distribution is learned $p(x, y)$. Recognizing the object is to estimate $p(y|x)$ (p of y given x). To estimate this posterior distribution, Gibbs sampling is used, by forward propagating x (with x_u randomly initialized) and backpropagating y alternatively, with the weight of network fixed. This gives the completed shape and prediction simultaneously. This procedure is run in parallel for a large number of times, and the class corresponds to the most frequently sampled class.


#### Notes
- For shape completion, more recent work from Charles Qi's team is [CNNComplete](https://arxiv.org/pdf/1612.00101.pdf).
- About next-best-view prediction, the equations in Section 4.2 is very hard to understand, but Fig. 4 is very straightforward.

