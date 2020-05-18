# Classical Keypoints

_May 2020_

tl;dr: A summary of classical keypoints and descriptors.

#### BRIEF
- [BRIEF: Binary Robust Independent Elementary Features](https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf) <kbd>ECCV 2010</kbd>
- The sampling pattern is randomly generated but [the same for all image patches](https://gilscvblog.com/2013/09/19/a-tutorial-on-binary-descriptors-part-2-the-brief-descriptor/#comment-328). In openCV, the [sampling sequence is pre-fixed](https://gilscvblog.com/2013/09/19/a-tutorial-on-binary-descriptors-part-2-the-brief-descriptor/#comment-1282).
	- patch window is 31 x 31.
- [blog review on BRIEF descriptor](https://gilscvblog.com/2013/09/19/a-tutorial-on-binary-descriptors-part-2-the-brief-descriptor/)

#### ORB
- [ORB: an efficient alternative to SIFT or SURF](http://www.willowgarage.com/sites/default/files/orb_final.pdf)  <kbd>ICCV 2011</kbd>
- Sampling pairs should have **uncorrelation and high variance** to ensure the fixed length would encode maximum discriminative information.
- ORB is improved BRIEF:
	- ORB uses an orientation compensation mechanism, making it rotation invariant.
	- ORB learns the optimal sampling pairs, whereas BRIEF uses randomly chosen sampling pairs.
- [blog review on ORB](https://gilscvblog.com/2013/10/04/a-tutorial-on-binary-descriptors-part-3-the-orb-descriptor/)

#### Key ideas
- Summaries of the key ideas

#### References
- [nterest Point Detector and Feature Descriptor Survey](https://core.ac.uk/download/pdf/81870989.pdf)