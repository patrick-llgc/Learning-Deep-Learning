# [OWOD: Towards Open World Object Detection](https://arxiv.org/abs/2103.02603) 

_April 2021_

tl;dr: Propose a new task of open world object detection. Detect unknown objects, while able to continuous benefit from new labels on these unknown objects. 

#### Overall impression
All existing object detectors have one strong assumption that all the classes are to be detected would be available at training phase. Although there are open set classification problem, the object detection is not studied. Object detectors are trained to detect unknown objects as background. 

The open world object detector is required to detect unknown classes as unknown, and is able to detect these classes when the unknown instances are forwarded to a human annotator to label, without being trained from scratch. 

#### Key ideas
- Contrastive loss (similar to push-pull loss) for each class and unknown class.
	- Each class has a prototype vector (class specific feature anchor). This vector is updated after the burn-in stage and updated with momentum. --> looks like the method in [MoCo](moco.md)
	- With the features learned with contrastive loss, we can tell known from unknown, by the **cls distribution of the logits**.
- Autolabeling with RPN. Regions with high objectiveness score but do not have overlap with any GT are labeled as unknown. 
- **Balanced finetuning**
	- How to learn new classes with a trained detector, data containing annotation of objects previously labeled as unknown. 
	- Train from scratch is impossible due to catestrophic forgetting. One way to **alleviate forgetting** is by storing few examples ($N_{ex}$ = 50) and replaying, a proven effective method in few shot object detection. See [FsDet](fsdet.md).

#### Technical details
- Evaluation metric on open world detection
	- Wilderness Impact (WI): P (eval on known class) / P (eval on unknown class) - 1. This characterizes the impact of the introduction of new classes
	- Absolute Open-Set Error (A-OSE): number of counts of unknown objects that get wrongly classified as known class.
- A sample qualitative result
![](https://camo.githubusercontent.com/48fec353f66f8528547527d6fc598e25cf6c16a4b919f554ed318ebd1ee438e0/68747470733a2f2f6a6f736570686b6a2e696e2f6173736574732f696d672f6f776f642f6578616d706c652e706e67)

#### Notes
- [code on github](https://github.com/JosephKJ/OWOD)

