# [Review of Image Descriptors](paper_notes/review_image_descriptors.md)

_November 2019_

This review is based on the pyimagesearch course.

#### Image Descriptors
Image descriptors and feature vectors quantify and abstractly represent an image using only a list of numbers.

- Feature vector: An abstraction of an image used to characterize and numerically quantify the contents of an image. Normally real, integer, or binary valued. Simply put, a feature vector is a list of numbers used to represent an image.
	- this feature can be passed down to image classifier or image search engine
- Image descriptor: An image descriptor is an algorithm and methodology that governs how an input image is **globally** quantified and returns a feature vector abstractly representing the image contents.
	- they tend to be much simpler than feature descriptors
	- HoG, LBPs, Harlick texture
- Feature descriptor: A feature descriptor is an algorithm and methodology that governs how an input region of an image is **locally** quantified. A feature descriptor accepts a single input image and returns **multiple feature vectors**.
	- SIFT, SURF, ORB, BRISK, BRIEF, and FREAK
	
### Local features
Keypoint detection and feature extraction: 
- keypoints are simply the (x, y)-coordinates of the interesting, salient regions of an image.
- feature extraction is the process of extracting multiple feature vectors, one for each keypoint
How to use multiple features per image? Keypoint matching or bag-of-visual-words.
	
#### FAST (Features from Accelerated Segment Test)
- Fast algorithm to detect corners
- [Fusing Points and Lines for High Performance Tracking](https://gurus.pyimagesearch.com/wp-content/uploads/2015/06/rosten_2005.pdf) <kbd>ICCV 2005</kbd>
- A test is performed for a feature at a pixel p by examining a circle of 16 pixels (a Bresenham circle of radius 3) surrounding p. A feature is detected at p if the intensities of at least 12 contiguous pixels are all above or all below the intensity of p by some threshold, t.
![](https://docs.opencv.org/3.4/fast_speedtest.jpg)
- [openCV example](https://docs.opencv.org/3.1.0/df/d0c/tutorial_py_fast.html#gsc.tab=0)

#### Harris
- Quite fast (although slower than FAST), more accurate

#### SIFT