# [DeepSignals: Predicting Intent of Drivers Through Visual Signals](https://arxiv.org/pdf/1905.01333.pdf)

_June 2019_

tl;dr: Detecting automotive signals with CNN+LSTM to decode driver intention.

#### Overall impression
Vehicle light detection is a rather overlooked field in autonomous driving, perhaps due to the lack of public datasets. As long as autonomous cars and human driver co-exist, the capability to decode human driver's intention through visual signal is important, for vehicle-to-vehicle communication.

The paper's performance is not that good. Perhaps due to the severe imbalance in the dataset.

#### Key ideas
- The use of attention is quite enlightening. This eliminates the need for turn signal light recognition.
- The study depend on a video of cropped patches, and trained on GT annotation. The performance degrades when sequence of real detection is used. (This might be improved via data augmentation during training.)

#### Technical details
- Annotation:
	- Intention/situation: left merge, right merge, emergency flashers, off, unknown (occluded), **brake**
	- left/right light: ON, OFF, unknown (occluded)
	- view: on, off, front, right
	- **brake lights: ON, OFF, unknown**
- More balanced datasets

#### Notes
- Q: why no brake light? This need to be added to the annotation
- Q: how to annotate unknown intention?
- Q: how to speed up annotation? Each vehicle is needed to assign a token throughout the video (uuid).
- Q: FP (off or unknown classified as any other state) is critical. We need this number as low as possible.
