# [BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270)

_March 2022_

tl;dr: Spatiotemporal transformer for BEV perception of both dynamic and static elements. 

#### Overall impression
BEVFormer exploits both spatial and temporal information by interacting with spatial and temporal space through predefined grid-shaped BEV queries. This includes a spatial cross-attention and a temporal self-attention. -->  actually also a type of cross-attention.

The paper believes the previous methods based on depth prediction ([Lift Splat Shoot](lift_splat_shoot.md), [CaDDN](caddn.md), [PseudoLidar](pseudo_lidar.md)) are subject to compounding errors, and thus favors direct method.

It significantly outperforms previous SOTA [DETR3D](detr3d.md) by 9 points in NDS score on nuScenes dataset.

The introduction of temporal module helps in 3 directions: 1) more accurate velocity estimation 2) more stable location and orientation 3) higher recall on heavily occluded objects. 

#### Key ideas
- The BEV perception of BEVFormer includes 3D object detection and map segmentation.
- Spatiotemporal transformer encoder that projects multicam video input to BEV representation.
- [Deformable DETR](deformable_detr.md) is adppted as efficient backbone and its formulation is a bit diff from global attention. DeformAttn(Q, K=p, V=input), where feature V at p is bilaterally sampled.
- Three tailored designs of BEVFormer: BEV queries, spatial cross-attention and temporal self-attention.
- BEV queries
	- HxWxC = 200x200x256, s grid size = 0.512 m. This corresponds to a grid = 50 m to all sides in nuScenes. The query responsible for located at p=(x, y) is $Q_p \in R^{1\times C}$.
- Spatial cross-attention (SCA)
	- For each pillar in the BEV grid, 4 points are sampled along diff height (every 2 m from -5 m to 3 m), and projected to diff images to form reference points. 
	- $SCA(Q_p, F_t)$ can be written as global attention $\text{Attn}(Q=Q, K=F_t, V=F_t)$. 
	- BEVFormer used Deformable attention, and therefore $\sum_i\sum_j^{N_{ref}}\text{DeformAttn}(Q_p, P(p, i, j), F_t^i)$. i iterates through all camera view, and j iterates through N-ref points in deformable attention. P(p, i, j) is in F_t space.
- Temporal self-attention
	- Align $B_{t-1}$ to Q at current timestamp with ego-motion, then denote as $B'_{t-1}$
	- $TSA(Q_p, {Q, B'_{t-1}})$ can be written as global attention $\text{Attn}(Q=Q, K=Q, V=Q)+ \text{Attn}(Q=Q, K=B'_{t-1}, V=B'_{t-1})$. **Essentially it is a cross-attention of BEV query of B_t-1 feature, but also enforced with a self-attention of Q on itself.**
	- BEVFormer used Deformable attention, and therefore $\text{DeformAttn}(Q_p, p, Q) + \text{DeformAttn}(Q_p, p, B'_{t-1})$. p is in BEV space, the same space as Q and B'_t-1.


#### Technical details
- This is the first work that leverages the temporal information to predict velocity, which got the SOTA results among all vision-based method. It reduced the velocity estimation error by almost 50%. It is unbelievable that previous work used single images to predict velocity.
- Map segmentation uses [Panoptic SegFormer](panoptic_segformer.md) head.
- Temporal training: 3+1 for the past 2 seconds. All but the last step require no gradient. --> This can help speed up training. Maybe we can do the same for convLSTM.
- There are 6 encoder layers, the output of each layer is used as the updated queries as the next layer. B_t-1 remain the same for each encoder layer and require no gradient.
- For nuscenes, HxWxC = 200x200x256, s grid size = 0.512 m, spanning 50 meters to all 4 directions. For waymo open dataset, HxWxC = 300x220x256, s grid size = 0.5 m, spanning almost 75 meters in all direction except the rear.
- *Negative transfer* in joint training: when two tasks regress each other in joint training.


#### Notes
- [github](https://github.com/zhiqi-li/BEVFormer) (will open source in 2022/06)
