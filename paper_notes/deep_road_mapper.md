# [DeepRoadMapper: Extracting Road Topology from Aerial Images](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.pdf)

_August 2020_

tl;dr: Extract road topology from satellite images.

#### Overall impression
This is one of the first paper on extracting road network based on aerial images captured by satellite. Note that this is not HD map as it does not contain lane level information.

The following work are focused on road network discovery and are NOT focused on HD maps.

- [DeepRoadMapper](deep_road_mapper.md): semantic segmentation
- [RoadTracer](road_tracer.md): like an DRL agent
- [PolyMapper](polymapper.md): iterate every vertices of a closed polygon

#### Key ideas
- Semantic segmentation
- Thinning
- Pruning small branches, closing small loops
- A* search algorithm for connecting disconnected roads. 

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

