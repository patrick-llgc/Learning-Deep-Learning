# [Deep Learning Based Rib Centerline Extraction and Labeling](https://arxiv.org/pdf/1809.07082)

_Feb 2019_

tl;dr: Locate the first and last pairs of ribs and use as anchor for labeling

#### Overall impression
The paper proposes a smart idea of classifying all 12 pairs of ribs into three classes: first, intermediate and last. The final results of ~80% success rate is a bit lacking though. Previous methods of detecting seed points and track has a disadvange that it often misses an entire rib if the seed is missing. This paper generates rib candidates first and then traced to generate centerline.

#### Key ideas
- A 3D F-Net (a lightweight U-Net) structure is used to perform semantic segmentation of all voxels into four classes, first ribs, last ribs, intermediate ribs and background.  
- This idea exploits the fact that the first pair of ribs and the last pair look distinctive from other pairs, and other pairs are hard to distinguish.
- Another advantage of this approach is that it is able to label ribs even if only the rib cage can be seen partially.
- The images are sparsely labeled. Only spline control points are annotated in each rib, then spline fitted and dilated to be the centerline. This dilated centerline volume is fed into the F-Net. 
- The tracing algorithm is quite esoteric and seems hard to parallelize. It deals with local drop-outs of the probability response. 
- The algorithm also first detects a rib cage bounding box to reduce false positive, as many CT scans extends beyond the rib cage.

#### Technical details
- F-Net is essentially an end-to-end network with image pyramid on the encoding path with the same decoding path as U-Net. The 3D image is isotropically resampled  (1.5 mm^3, 3 mm^3, 6 mm^3).

#### Notes
- The idea of identifying the first and last pair of ribs can be applied on top of rib candidates (without any subclasses, this can be done using CV methods).. This may leads to better results.
- Rib cage bounding box generation is essentially scouting. This should be very helpful in building a robust system.

