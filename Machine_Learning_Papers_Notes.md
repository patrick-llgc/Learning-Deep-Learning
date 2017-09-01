# Machine Learning Papers Notes
### Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation ([link](https://arxiv.org/pdf/1609.08144.pdf))
- Three inherent weaknesses of Neural Machine Translation (that prohibited NMT to overtake phrase based machine translation):
	1. Slower training and inference speed;
		- To improve training time: GNMT is based on LSTM RNNs, which have 8 layers with residual connections between the layers to encourage gradient flow. To improve inference time, low-precision arithmetic are used, further accelerated by google's TPU.
	2. ineffectiveness in dealing with rare words;  
		- To effectively deal with rare words: sub-word units ("wordpieces") were used for inputs and outputs.
	3. failure to translate all words in the source.
		- To translate all of the provided input, a beam search technique and a coverage penalty are used.
- Phrase-based machine translation (PBMT), as a type pf statistical machine translation method, has dominated machine translation for decades. NMT has been used as part of the PBMT and achieve promising results, but end-to-end learning based on NMT for machine translation has only started to surpass PBMT recently.
	- attention mechanism, character decoder, character encoder, subword units have been proposed to deal with rare words.
- GNMT is a sequence-to-sequence learning framework with attention. In order to achieve high accuracy, GNMT has to have deep enough encoder and decoder to capture subtle irregularities in the source and target.
- TBC


### DeepEM3D: Approaching human-level performance on 3D anisotropic EM image segmentation [link](https://academic.oup.com/bioinformatics/article-abstract/33/16/2555/3096435/DeepEM3D-approaching-human-level-performance-on-3D?redirectedFrom=fulltext)


### Sensor fusion [link](https://www.youtube.com/watch?v=xDDN8Q0hJos)
- 2 approchaes to sensor fusion
	- fuse input data from sensors before analysis
	- fuse analysis output data
- Prerequisites of sensor fusion
	- sensor synchronization using GPS
	- Localization in 6D
		- GPS is not reliable or accurate in urban canyons


