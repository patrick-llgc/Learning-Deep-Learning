# [TW-SMNet: Deep Multitask Learning of Tele-Wide Stereo Matching](https://arxiv.org/abs/1906.04463)

_November 2019_

tl;dr: Depth estimation by fusing the images from two cameras with different focal length.

#### Overall impression
The paper is among the first to fuse stereo pairs with different focal length. For the intersection FoV (tele FoV), we can resize the wide FoV image to match that of the tele FoV, reducing the problem to normal stereo pair matching algorithm. 

#### Key ideas
- Depth prediction via stereo matching on the intersected FoV improves when fusing stereo info 
- Single image depth estimation on the wide FoV has better performance on the periphery, but not so much in the overlapped FoV.
- TW-SMNet merges the depth of the two. The single image depth estimation branch forces the network to learn semantics. Actually only the stereo matching prediction is used during inference. **The single image depth estimation is used as auxiliary training branch**.
- Proper fusion of the two predictions can also improve performance (the paper has a long discussion on how to fuse them)
	- input fusion: the authors fused the input from the initial results (absolute metric value) from the stereo matching in tele FoV to the wide FoV raw image. This idea is similar to the [sparse to dense](sparse_to_dense.md).
	- output fusion: pixel-wise decision selection. --> This leads to abrupt change in depths. Need to use global smoother such as FGS (Fast global smoother).
	- deep fusion of depth uses robust regression as second stage refinement.

#### Technical details
- **classification-based robust regression** loss, by classifying regression target range into bins, then predict. Note that no cross entropy loss is added. The loss is on the soft prediction (weighted average of bin centers by the scores past softmax) --> this is very similar to the multi-bin loss proposed by [deep3dbox](deep3dbox.md).
- The stereo matching uses PSM-Net (CVPR 2018), and monocular depth estimation uses DORN (CVPR 2018)
- Two TW-SMNet is trained for tele lens (T) and wide lens (W). For T model, wide FoV is cropped. For W model, tele FoV is padded.

#### Notes
- Kitti's stereo pairs has a baseline of 54 cm. Human has baseline of 6 cm. Most trifocal lens system on the market has a couple of cm, smaller than human eye, and thus not much disparity.
- Note on the results: merging the two actually finds the middle ground between the TW-SMNet models T and W. With stereo info the intersected FoV has much better depth estimation than single image based model.
- Publication at ICIP 2019 [Multi-Task Learning of Depth from Tele and Wide Stereo Image Pairs](https://ieeexplore.ieee.org/abstract/document/8803566) <kbd>ICIP 2019</kbd>


