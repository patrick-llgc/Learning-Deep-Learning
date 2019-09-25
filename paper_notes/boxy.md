# [Boxy Vehicle Detection in Large Images](https://boxy-dataset.com/boxy/index)

_September 2019_

tl;dr: A large dataset with 3D-like labels from Bosch.

#### Overall impression
The author proposed to annotate cuboids with 2 plans, one axis-aligned bounding box (AABB) for rear side and the other trapezoid for the side. This annotation idea is brilliant. The authors did mention the **upper front** point is ambiguous to label. 

Another big dataset with 3D-like label is [BoxCars](https://github.com/JakubSochor/BoxCars), but BoxCars is a surveillance dataset and the vehicle angle is different. (For example, in surveillance, we could see the top of the car in most images, but in almost none of the autonomous driving scenes).

#### Key ideas
- Color information from camera is still quite important, such as brake lights and turn signals which are not available in other sensors. 

#### Technical details
- The dataset is quite large, at 1.1 TB! 
- Each image is at 5 Mega Pixels, 2464x2056, almost 1x1 aspect ratio. ~2 M annotations. 
- The majority of the annotation is below 50x50 pixels.
- Number of vehicles occupying each pixels --> this heat map can guide the placement of anchors. The majority is cluttered around vanishing point, and neighboring lanes. 
- **The rear bbox should cintain the complete rear without containing the side or front mirrors**. 
- The annotation has 7 degrees of freedom. Two terms in loss, one is cross entropy (visibility classification) and the other for regression.

#### Notes
- The majority of label annotation firm offer 3D cuboid annotation service, such as [scale.ai](https://scale.com/blog/3d-cuboids-annotations) and [playment](https://www.youtube.com/watch?v=KWyFnXKvBCc).
- This 