# [The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation](https://arxiv.org/abs/1904.03215)

_December 2019_

tl;dr: A new benchmark measuring how well methods detect potentially hazardous anomalies in driving scenes. 

#### Overall impression
Embeddings of intermediate layers hold important information for anomaly detection.


#### Key ideas
- Bayesian DL: epistemic uncertainty, aleatoric uncertainty, distributional uncertainty 
- Novelty detection (Out of distribution detection): one class cls which aim at discriminative embeddings, density estimations, and generative reconstruction.
- Softmax score is not a reliable score for anomaly detection
- Most better performing methods require special loss that reduced segmentation accuracy (tradeoff between better outlier detection and error. Cf tradeoff between better uncertainty calib and error)
- Learning anomaly detection from fixed OoD data is on par with unsupervised methods for most of the datasets. **Void classifier is most practical way forward**. A separate void class is concisely better than maximizing the softmax entropy. A separate void class is also most practical. 

#### Technical details
- Lost & Found dataset is real dataset, it can be used to compare the performance on synthetic datasets to identify methods that detect image in-painting instead of anomalies. 
- Image inpainting postprocessing steps please refer to [Augmented Reality Meets Computer Vision : Efficient Data Generation for Urban Driving Scenes](https://arxiv.org/abs/1708.01566) <kbd>IJCV 2018</kbd> (data augmentation with AR)
- evaluation scores: AP, and FPR@0.95Recall

#### Notes
- Foggy cityscape: dataset with adjustable visibilities. 
- [WildDash](http://openaccess.thecvf.com/content_ECCV_2018/papers/Oliver_Zendel_WildDash_-_Creating_ECCV_2018_paper.pdf) <kbd>ECCV 2018</kbd> and [RobustVisionChallenge](http://www.robustvision.net/workshop.php) <kbd>CVPR 2018</kbd> for semantic and instance segmentation
- [Safe Visual Navigation via Deep Learning and Novelty Detection](http://www.roboticsproceedings.org/rss13/p64.pdf) (generative reconstruction)
