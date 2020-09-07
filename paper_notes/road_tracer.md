# [RoadTracer: Automatic Extraction of Road Networks from Aerial Images](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bastani_RoadTracer_Automatic_Extraction_CVPR_2018_paper.pdf)

_August 2020_

tl;dr: Dynamic training of a CNN as an DRL agent to draw maps. 

#### Overall impression
The following work are focused on road network discovery and are NOT focused on HD maps.

- [DeepRoadMapper](deep_road_mapper.md): semantic segmentation
- [RoadTracer](road_tracer.md): like an DRL agent
- [PolyMapper](polymapper): iterate every vertices of a closed polygon

RoadTracer noted the semantic segmentation results are not a reliable foundation to extract road networks. Instead, it uses an iterative graph construction to get the topology of the road directly, avoiding unreliable intermediate representations. 

The network needs to make a decision to step a certain distance toward a certain direction, resembling an agent in a reinforcement learning setting. This is somehow similar to the cSnake idea in [Deep Boundary Extractor](deep_boundary_extractor.md).

#### Key ideas

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

