# [Stitcher: Feedback-driven Data Provider for Object Detection](https://arxiv.org/abs/2004.12432)

_May 2020_

tl;dr: Data augmentation method to boost small object detection.

#### Overall impression
Stitcher proposed two methods to stitch multiple images into one, either along spatial dimension or along channel dimension. The first one is similar to the mosaic data aug in [yolov4](yolov4.md).

#### Key ideas
- Hypothesis: small objects contribute to less loss and thus receives less supervision.
- Solution: if small object loss is below a certain threshold in a certain iteration, in the next iteration, use stitcher dataloader to resize and tile images into 4x4 format.
- The alternative solution of stitching into batches is similar to dynamic shape training in [yolov4](yolov4.md) and [multigrid training for video](multigrid_training.md).
- Setting to use stitcher all the time hurt the performance (input resolution matters!). Using stitcher about 10% of the time helps. 
- Use stitcher randomly also helps with the performance.

#### Technical details
- objects smaller than 32x32 is called small in COCO.
- **Nearest neighbor downsampling** in stitcher.
- Stitcher is superior to mixup, SNIP and SNIPER.
- Stitcher helps with curbing overfitting. Conventional setting overfits when using 6x training schedule, but stitcher still improves.
- - Threshold (small obj loss / total loss = 0.1) works best. Threshold = 1 is essentially selecting stitched images all the time and leads to worse performance.

#### Notes
- Questions and notes on how to improve/revise the current work  

