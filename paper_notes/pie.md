# [PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf)

_June 2020_

tl;dr: Pedestrian intension estimation dataset and achieves ~80% of accuracy and ~90% F1 score.

#### Overall impression
Intention and trajectory prediction are different. Trajectory prediction are only effective when the pedestrians have started crossing or about to do so, so basically the algorithm **react** to an action instead of **anticipating** it.

Intention helps trajectory prediction. This is similar conclusion as in [IntentNet](intentnet.md).

The key question is "does this pedestrian **want** to cross the street?"

#### Key ideas
- Intention estimation allows one to predict a future situation using expected behaviors rather than merely reply on scene dynamics.
- Annotation:
	- Labels walking, standing, looking, not looking, crossing, not crossing.
	- Crossing intention confidence from 1 to 5. 
	- Bbox with occlusion level.
- Annotators are allowed to show a clip ~3 sec long before the vehicle reaches 1.5-3 sec time to event, so the labeler does not see the final event. 
- Inter-rater consistency is high. 
- Humans are good at telling if a pedestrian is going to cross. There are 2/1800 (0.1%) samples that crossed the street but the annotators indicate otherwise. 
- LSTM is used for intention and trajectory prediction.

#### Technical details
- Uses AMT to gather 10 annotations per video clip. 
- Feeding bbox and image patches around the bbox (context info) helps. 

#### Notes
- It would have been very interesting to know the average performance of a human annotator. 

