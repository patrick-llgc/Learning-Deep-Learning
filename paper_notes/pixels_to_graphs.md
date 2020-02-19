# [Pixels to Graphs by Associative Embedding](https://arxiv.org/abs/1706.07365)

_January 2020_

tl;dr: Generate relationship graphs based on associative embedding.

#### Overall impression
This paper is based on and is almost concurrent with [associative embedding](associative_embedding.md).

The heatmap for objects already has some preliminary structure for and may have inspired [CenterNet](centernet_ut.md).

#### Key ideas
- Two heatmaps to predict vertex (detections) and edges (relations).
	- Individual feature vectors are extracted from top heatmap locations.
	- Feed into fc layers to predict vertex properties (class ID, bbx, ID) and edge properties (class ID, src ID, dst ID).
- The graph is a directed graph as many relations are directed (larger than, on top of, etc)
- Loss: on loss 
	- pull: the embeddings of one vertex and all edges leading to it. L2 loss.
	- push: hinge loss (changed from gaussian loss to improve convergence)


#### Technical details
- Output resolution: Different detection can share the same pixel. The higher output resolution, the smaller the chance of center collision.
- The output dimension d increased to 8 from 1 in original AE paper. This improves convergence
- The feature vector from one pixel can generate multiple object ID/prediction/embeddings.

#### Notes
- Questions and notes on how to improve/revise the current work  

