# [A study on Radar Target Detection Based on Deep Neural Networks](https://www.researchgate.net/publication/330748053_A_Study_on_Radar_Target_Detection_Based_on_Deep_Neural_Networks)

_July 2019_

tl;dr: Detection on RD map. Use CNN to replace CFAR (adaptive thresholding) and binary integration (time sequence processing).

#### Overall impression
The paper showed that there are minimum gain of CNN over CFAR under different noise levels. And it is better than binary integration for time sequence processing.

#### Technical details
- 61 frames, 0.5 ms frame time, 2 us per chirp.
- Seems like only one object in the frame.