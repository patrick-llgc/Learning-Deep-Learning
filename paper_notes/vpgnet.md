# [VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition](https://arxiv.org/abs/1710.06288)

_March 2020_

tl;dr: Use VP to guide lane and RSM (road surface marking) detection under adverse weather condition.

The design of catch-all channel dustbin/absence channel to recalibrate softmax heatmap is interesting, and both [SuperPoint](super_point.md) and [VPGNet](vpgnet.md) used the same trick.


#### Overall impression
This is the first paper on laneline and road marking detection under adverse weather. Under good weather, the performance improvement of VP prediction is not huge. Under adverse weather, it helps quite a lot.

The method requires extensive postprocessing and does not seem to be robust enough for industry use.

VP prediction only helps:

- Under adverse weather
- When trained first then used as pretraining to regress other tasks

#### Key ideas
- VP detection: three levels of difficulty. 
	- Easy: clear scene
	- Hard: a cluttered scene (traffic jam)
	- None: no vanishing point (intersection)
- VP regression:
	- Binary FG/BG classification does not yield good results due to class imbalance. --> maybe use focal loss and gaussian blurred GT?
	- **quadrant masks** separated by VP, and one catch-all "absence" channel. If there is no point in the image, then encourage every one to go to the absence channel.
- Lane and marking: 8x8 grid 
- Lane Postprocessing:
	- adaptive thresholding heatmap to get candidates
	- IPM to BEV
	- binning to cluster
	- Quadratic regression
- Road Marking
	- Select cells with high confidence
	- Merging
- VP postprocessing:
	- looking at absence channel
	- Find points where the confidence from each quadrant channel is close
- Multi-task learning (especially regressing the VP) improves the accuracy of lane lines.  If we use more tasks, more neurons respond, especially around the boundaries of roadways.

#### Technical details
- Balancing loss: set all weights to 1 first, then observe loss magnitude. Set w to be reciprocal of the magnitude. Rebalance when the loss magnitude from different tasks are very different.
- 2 Training stages: 
	- 1st stage only VPP (first look at vanishing point)
	- 2nd stage: train all other tasks: VPP get improved as well
	- if trained together, VPP become dependent on LLD. Not too much improvement

#### Notes
- Why not directly regress VP in the image as x and y?

