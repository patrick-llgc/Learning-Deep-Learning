# [PYVA: Projecting Your View Attentively: Monocular Road Scene Layout Estimation via Cross-view Transformation](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_Projecting_Your_View_Attentively_Monocular_Road_Scene_Layout_Estimation_via_CVPR_2021_paper.html)

_September 2021_

tl;dr: Transformers to lift image to BEV.

#### Overall impression
This paper uses a cross-attention transformer structure (although they did not spell that out explicitly) to lift image features to BEV and perform road layout and vehicle segmentation on it.

It is difficult for CNN to fit a view projection model due to the locally confined receptive fields of convolutional layers. Transformers are more suitable to do this job due to the global attention mechanism.

Road layout provides the crucial context information to infer the position and orientation of vehicles. The paper introduces a context-awre discriminator loss to refine the results. 

#### Key ideas
- CVP (cycled view projection)
	- 2-layer MLP to project image feature X to BEV feature X', following [VPN](vpn.md)
	- Add cycle consistency loss to ensure the X' captures most information
- CVT (cross view transformer)
	- X' as Query, X/X'' as key/value
- Context-aware Discriminator. This follows [MonoLayout](monolayout.md) but takes it one step further. 
	- distinguish predicted and gt vechiles
	- distinguish predicted and gt correlation between vehicle and road

#### Technical details
- Summary of technical details

#### Notes
- [code on Github](https://github.com/JonDoe-297/cross-view)

