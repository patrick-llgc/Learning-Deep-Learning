# [PolyMapper: Topological Map Extraction From Overhead Images](https://arxiv.org/abs/1812.01497)

_August 2020_

tl;dr: Map buildings and roads as polygon.

#### Overall impression
Identify keypoints first, then starting with one arbitrary one vertex, connect them according to the Left/Right hand rule (or [Maze solving algorithm](https://en.wikipedia.org/wiki/Maze_solving_algorithm)), then there is one unique way to define the graph.

The following work are focused on road network discovery and are NOT focused on HD maps.

- [DeepRoadMapper](deep_road_mapper.md): semantic segmentation
- [RoadTracer](road_tracer.md): like an DRL agent
- [PolyMapper](polymapper): iterate every vertices of a closed polygon

[Polyline loss](polyline_loss.md) and [DAGMapper](dagmapper.md) focuses on HD mapping tasks with lane-level information. 

Road network extraction is still very helpful for routing purposes, but lack the fine detail and accuracy needed for a safe localization and motion planning of an autonomous car. 

#### Key ideas
- Find RoI with RPN
- Identify keypoints
- Connect keypoints with RNN (Conv-LSTM)

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

