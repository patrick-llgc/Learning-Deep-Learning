# [How to Keep HD Maps for Automated Driving Up To Date](http://www.lewissoft.com/pdf/ICRA2020/1484.pdf)

_December 2020_

tl;dr: A linklet based updating method for HD maps.

#### Overall impression
The authors believed that [feature based mapping](lifelong_feature_mapping_google.md) which maintains the persistence score of each feature does not allow large scale application.

The central idea is to use fleet data to estimate probability of change and update the map where needed.

#### Key ideas
- Partition the map into linklets in the topology graph defined in SD map. 
- Three steps:
	- Change detection: detecting if a change in the map has occurred (difference between perception and map). This score is predicted by a gradient boosted tree. This updates the is_changed function F for traversed linklet. 
	- Job creation: when the aggregated linklet change probability is larger than a certain threshold (also learned), trigger map updating.
	- Map updating: 

#### Technical details
- Summary of technical details

#### Notes
- [TomTom](https://www.tomtom.com/blog/maps/continuous-map-processing/) has a similar method to detect changes and update map patches. 

