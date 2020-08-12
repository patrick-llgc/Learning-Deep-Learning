# [PolyMapper: Topological Map Extraction From Overhead Images](https://arxiv.org/abs/1812.01497)

_August 2020_

tl;dr: Map buildings and roads as polygon.

#### Overall impression
Identify keypoints first, then starting with one arbitrary one vertex, connect them according to the Left/Right hand rule (or [Maze solving algorithm](https://en.wikipedia.org/wiki/Maze_solving_algorithm)), then there is one unique way to define the graph.

#### Key ideas
- Find RoI with RPN
- Identify keypoints
- Connect keypoints with RNN (Conv-LSTM)

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

