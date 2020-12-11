# [Towards lifelong feature-based mapping in semi-static environments](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43966.pdf)

_December 2020_

tl;dr: Feature persistence model to keep features in the map up to date.

#### Overall impression
Vanilla SLAM assumes a static world. They have to adapt in order to achieve **persistent autonomy**. This study proposed a **feature persistent model** that is based on survival analysis. It uses a recursive Bayesian estimator (persistence filter).

In summary, any observation existence boosts the existence confidence, any observation of absence degrades existence conf, and lack of observation decays existence conf.

This method has a good formulation but seems to be a bit heavy and does not allow large scale application. See [Keep HD map updated](keep_hd_maps_updated_bmw.md).

#### Key ideas
- Feature based mapping views the world as a collection of features (lines, planes, objects, or other visual interest points). Mapping is then identify and estimate their state (position, orientation, color).
	- In semi-static mapping we have to both add new features to the map and remove existing features from the map. 
	- The detector is not perfect as well, so it is insufficient if a feature is still present or not. We can only update the **belief**.
	- The passage of time matters. An observation 5 min ago should be different from one observation 5 days ago.

#### Technical details
- Summary of technical details

#### Notes
- Questions and notes on how to improve/revise the current work  

