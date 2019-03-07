# [Volumetric and Multi-View CNNs for Object Classification on 3D Data](https://arxiv.org/pdf/1604.03265.pdf)

_Mar 2019_

tl;dr: Improvement of volumetric CNNs (3d shapenets) closes its gap with multi-view CNNs (MVCNN). 

#### Overall impression
The paper starts with one hypothesis: the performance gap of volumetric and multi-view CNN is due to the resolution difference. However experiment shows this only explains part of the gap. The paper then takes on two directions: improve the volumetric CNN architecture, and exploit the resolution in MVCNN. This paper already shows the concise and straightforward style of Charles Qi's style later shown in [pointnet](pointnet.md).

#### Key ideas
- The gap between 2D and 3D can be attributed to 2 factors: input resolution and network architecture. 
- The experiment to examine the effect of spatial resolution in 2D and 3D CNNs is done by sphere rendering. Spheres are used for discretization as they are view invariant.
- Two improved volumetric CNN are proposed
	- Sub volume supervision (SubvolSup): the 3D network **overfits** severely. To address this, more difficult but highly relevant auxiliary tasks are added to perform classification with partial volume (specifically with feature maps within each octant).
	- Anisotropic probing (AniProbing): use an anisotropic kernel to mimic 2D projection of a 3D input. 
- Use of orientation-pooling with multi-orientation (MO) input augmentation to boost the performance of SubvolSup.
	- As expected, AniProbing benefits more from the augmentation. In other words, AniProbing is inspired by MVCNN and is supposed to use with multi-orientation augmentation.
- Another way to boost the performance of volumetric CNN is to use a [spatial transformation network](paper_notes/stn.md). STN tends to align all 3D volumes to a canonical viewpoint.

#### Technical details
- AniProbing is different from 2D rendering of a 3D object with computer graphics in two ways: it "sees through" the 3D object and provides an x-ray like scanning capacity; it saves computation time. 

#### Notes
- [Video presentation](https://www.youtube.com/watch?v=bE7jzHJiQWw) at CVPR 2015.
- How about using deterministic average or max pooling, instead of learning an anisotropic kernel in AniProbing?
