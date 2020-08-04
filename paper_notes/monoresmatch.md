# [MonoResMatch: Learning monocular depth estimation infusing traditional stereo knowledge](https://arxiv.org/abs/1904.04144)

_August 2020_

tl;dr: Inject proxy labels from traditional stereo knowledge into monodepth with stereo.

#### Overall impression
The paper learns to emulate a binocular setup. Basically it hallucinates a disparity map (DispNet), refines it with geometric constraints (with horizontal correlation layer), and then estimates the depth. It it similar to and inspired by [Single View Stereo Matching]().

The paper is similar to [Depth Hints](depth_hints.md) in the sense that it also uses proxy labels from SGM (semi-global matching) on stereo pairs (unavailable during inference time) to guide monodepth pipeline, but it adds the proxy label self check to reduce the noise. 

Both [Depth Hints](depth_hints.md) and [MonoResMatch](monoresmatch.md) propose to use cheap stereo GT to build up monodepth dataset. [Depth Hints](depth_hints.md) uses multiple param setup to obtain an averaged proxy label and use a soft (hint) supervision scheme. [MonoResMatch](monoresmatch.md) uses left-right consistency check to filter out spurious predictions and a traditional hard supervision scheme. 

#### Key ideas
- How to perform proxy label distillation
	- Learn conf measure together with noisy label
	- Perform left/right consistency check to remove inconsistent labels.

#### Technical details
- Summary of technical details

#### Notes
- The output of DispNet is the feature map of left/right views.
- Why use proxy labels but not directly use GT? We don't have stereo input images anyway. --> maybe this reduces cost to build GT systems significantly.