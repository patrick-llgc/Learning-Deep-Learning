# [MT-CNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment)

_September 2019_

tl;dr: One of the most widely used method for face detection and face landmark regression.

#### Overall impression
The paper seems rather primitive compared to general object detection frameworks like faster rcnn. MTCNN is more like the original rcnn method. 

However it is also enlightening that a very shallow CNN (O-Net) applied on top of cropped image patches can regress landmark accurately. Landmark regression given an object bbox may not require that large of a receptive field anyway.

The paper is largely inspired by Hua Gang's paper [cascnn: A Convolutional Neural Network Cascade for Face Detection](http://users.eecs.northwestern.edu/~xsh835/assets/cvpr2015_cascnn.pdf).

#### Key ideas
- Three stages
	- P-Net: proposal network on 12x12 input size
	- R-Net: FP reduction on 24x24 input size
	- O-Net: landmark regression on 48x48 input size
- P-Net is trained on patches but deployed convolutionally for detection. (or equivalently in a sliding window fashion)
- R-Net input is obtained from the output of P-Net
- O-Net input is obtained from the output of R-Net
- Multi dataset used differently
- Loss weighed differently and masked differently in different stages

#### Technical details
- Not a single model, but training can be done jointly. 


#### Notes
- [new implementation in TF](https://github.com/ipazc/mtcnn) and [original version](https://github.com/davidsandberg/facenet/tree/master/src/align)

