# [Associative Embedding: End-to-End Learning for Joint Detection and Grouping](https://arxiv.org/abs/1611.05424)

_January 2020_

tl;dr: Learn keypoints detection and association at the same time.

#### Overall impression
This is the foundation of [CornerNet](cornernet.md) which ignited a new wave of anchor less single-stage object detector in 2019.

The figure showcasing the 1D embedded results is really cool. This shows the **trivial task of grouping** after the learning.
![](https://pic1.zhimg.com/80/v2-71685932ee2ef3cd8da83a11f8de390c_hd.jpg)

The separation between different object in the tagging heatmap is also very impressive.
![](https://pic2.zhimg.com/80/v2-dc684098a4c31847f99e0c5c67440645_hd.jpg)

It can be also used for instance segmentation (and perhaps for tracking as well). Basically any CV problem that can be viewed as joint detection and grouping can benefit from associative embedding.


#### Key ideas
- Many CV task can be seen as joint detection and grouping (including object detection, as demonstrated by ConerNet later on).
- The output from the network is a detection heatmap and tagging heatmap. The embeddings serve as tags that encode grouping.
	- In the detection heatmap, multiple people should have multiple peaks. 
	- In the tagging heatmap, what matters is not the particular tag values, only the differences between them. 
	- If a person has m keypoints, the network will output 2*m heatmaps.
- **Dimension of embedding**: The authors argue that it is not important. If a network can successfully predict high-dimensional embeddings to separate the detections into groups, it should also be able to learn to project those high-dimensional embeddings to lower dimensions, as long as there is enough network capacity.
- Loss: **Tags within a person should be the same, and tags across people should be different.** Let h be tag value, T = {x_nk} is gt keypoint location
	- reference embedding (average embedding of one object): $\bar{h}_n = \frac{1}{K} \sum_k h_k (x_{nk})$
	- pulling force for each person: $L_g(h, T)_{inner} = \frac{1}{N} \sum_n \sum_k (\bar{h}_n - h_k(x_{n,k}))^2$ 
	- pushing force for different person: $L_g(h, T)_{outer} = \frac{1}{N^2} \sum_n \sum_{n'} exp{-\frac{1}{2\sigma^2} (\bar{h}_n - \bar{h}_{n'})^2}$ 
- Inference: max matching by both tag distance and detection score.

#### Technical details
- The system works best if objects are about the same scale. The paper used multiple scales during test time. The heatmaps are averaged, and the embedding are concatenated to a vector. Then compare vectors to determine groups.

#### Notes
- [Github code](https://github.com/princeton-vl/pose-ae-train) in pytorch

