# [Online Video Object Detection using Association LSTM](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lu__Online_Video_ICCV_2017_paper.pdf) 

_January 2020_

tl;dr: Online object detector based on video.

#### Overall impression
RNN is used for sequence learning, but RNN for video object detection is a harder problem. 

- It should capture multiple objects at the same time, where the number of objects varies from frame to frame. 
- Second, how to associate object in the RNN structure across multiple frames is a challenging problem

#### Key ideas
- t x N x D:
	- t frames
	- N detections at most. 
	- D = c + 4 + s x s is the feature length for each detected object. s x s feature map from ROI pooling. --> this may be replaced by 1x1 features from YOLO/SSD?
- Loss
	- Detection/classification loss (**Note**: only calculated for the last frame. Thus this useful for weakly supervised data as well)
	- Smooth loss: neighboring frames should have similar embedding vectors
	- Association loss: 
$$ L_{asso} = \sum_t \sum_{i,j} \theta_{ji} |\phi_{t-1}^i \phi_{t}^j| $$
	

#### Technical details
- MOTA challenge KPIs focus on tracking performance instead of detection performance.

#### Notes
- **The association loss does not make sense to me.** First the $\theta$ is not predicted but rather calculated (most likely in a Hungarian matching style). We want to encourage the similarity of the embedding of the same object across frames, but this will increase the loss. So there may be a minus sign in front of it. But then even that does not make sense either as nothing will stop the loss from learning all the embeddings to be exactly the same. 
- How to use rule-based algorithm to bootstrap deep learning? We can run rule-based algorithm twice, once with strict criterion (high precision) for positive case selection, and once with loose criterion (low precision) for negative case selection. The difference between the two runs are marked as "dont care".
- In YOLO, each cell in the feature map is a cheap version of ROI pooling, as it is used to regress bbox, so it should contain information to generate a discriminative embedding (association feature). It is similar to the idea of the heatmap in CenterNet. 

