# [MfS: Learning Stereo from Single Images](https://arxiv.org/abs/2008.01484)

_August 2020_

tl;dr: Mono for stereo. Learn stereo matching with monocular images.

#### Overall impression
The basic idea is to generate stereo training pair with mono depth to train stereo matching algorithms. This idea is very similar to that of Homographic Adaptation in [SuperPoint](superpoint.md), in that both generates training data and GT with known geometric transformation.

This still need a stereo pair as input during inference time. The main idea is to use monodepth to predict a depth map, sharpen it, and generate a stereo pair, with known stereo matching GT.

#### Key ideas
- Summaries of the key ideas

#### Technical details
- Training uses PSMNet (pyramid stereo matching)

#### Notes
- [Code on github](https://github.com/nianticlabs/stereo-from-mono/)

