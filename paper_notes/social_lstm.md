# [Social LSTM: Human Trajectory Prediction in Crowded Spaces](http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf)

_January 2020_

tl;dr: Summary of the main idea.

#### Overall impression
Adds a social pooling layer that pools the hidden stages of the neighbors within a spatial radius.

#### Key ideas
- Instead of a spatial occupancy grid, replace the occupancy with LSTM embedding.

#### Technical details
- [Social LSTM](social_lstm.md) is actually done from a surveillance view point (between perspective onboard cameras and BEV).

#### Notes
- [talk at CVPR](https://www.youtube.com/watch?v=q7LjIcKluK4): the animation of predicting a person passing through the gap of a crowd is cool.

