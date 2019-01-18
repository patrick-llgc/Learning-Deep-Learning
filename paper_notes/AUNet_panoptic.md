# [Attention-guided Unified Network for Panoptic Segmentation](https://arxiv.org/pdf/1812.03904.pdf)

_Jan 2019_

tl;dr: Exploit FG instance masks to refine BG stuff masks.

#### Overall impression
Using FG masks to refine BG masks is a natural idea as they are complementary to each other (the definition of panoptic segmentation indicates the mutual exclusiveness of BG and FG). The authors poured a lot of effort of coming up with the RoIUpsample layer to exploit the  masks.

#### Key ideas
* The paper uses two attention modules to exploit the feature maps and the masks from coarse to fine to compliment the semantic segmentation path.
* PAM (proposal attention module): coarse attention operation between the i-th scale BG feature map with the corresponding RPN feature map.
  * **For each scale in the feature pyramid**, S = S * (1-Sigmoid(Conv(P))) + S, where S is the i-th scal BG, and P is the i-th scale RPN.
  * An a dditional BG select function is used to fine tune the performance. It is essentially an SE-Net (squeeze and excitation net).
* MAM (mask attention module): predicted coarse areas from RPN branch lack enough cues for precise BG representation, and MAM is used to exploit precise instance masks
  * RoIUpsample reverts the RoIAlign operation, pasting the 14x14 or 28x28 mask back to the full resolution feature map. (the paper used two step process to paste to the full resolution)
  * The MAM module also uses an SE-Net like BG select function. 
* Inference: uses similar way of an NMS-like procedure to reconcile overlap like the panoptic paper, but relationship among categories are also considered during this procedure. 

#### Notes
* The information only flows from the instance branch to the semantic branch. Maybe introducing information flow from semantic to instance also boost the performance?
* It is not clear how much performance gain is obtained from exploiting relationship among categories during resolvoing overlapping pixels, as compared to the vanilla panoptic approach.
* From the ablation study, PAM seems to account for the majority of performance gain compared to the vanilla end-to-end training (similar to panoptic FPN). This is good news since the overwhelmingly engineered RoIUpsample would be hard to reimplement. This also means perhaps the majority of errors in isolated semantic segmentation and instance segmentation lies in the mis-classification of large segments (addressed by PAM), instead of refining the boundary (addressed by MAM).
