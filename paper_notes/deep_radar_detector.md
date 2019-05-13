# - [Deep Radar Detector](../assets/papers/deep_radar_detector.pdf) 

_May 2019_

tl;dr: Use the range-doppler-channel data cube to perform 4D detection task (range, doppler, azimuth and elevation).

#### Overall impression
This is a splendid paper with great overall introduction to conventional radar detection algorithms and how they replaced conventional blocks with CNN. This work also used radar calibration data with phase shift augmentation to train NN. 

With enough data and richer annotation, this work could be extended to detect multiple objects, and maybe even regress the size of the object, if the resolution is sufficiently high. 

#### Background
- The main goal of imaging radars is to create a relatively dense point cloud (less dense than lidar) of the vehicle surrounding at ta lower cost and with a superior weather immunity compared to optical lidars.
- Radar point cloud is not as dense as lidar point cloud, and it is subject to view point and temporal fluctuation. Two reasons: statistical nature of radar sensor, and conventional radar signal processing techniques.
- Conventional radar signal processing blocks: ADC (sampled radar echos) --> 2D-FFT --> Detector --> Beamforming --> Clustering/Tracking/Object Detection/Classification
-  The quality of the radar point cloud is mainly determined by detector and beamforming.
-  Direction of Arrival (DOA) is obtained through beamforming. One commonly used technique for MIMO radar is Bartlett and minimum variance distortion less response (MVDR) algorithm.
-  Many hand crafte parameters: threshold, margin, sizes and shapes. DL aims to eliminate the need for accurate parameter selection.
-  Calibration of radar is obtained with anechoic chamber.

#### Key ideas
- The method takes in complex (real and imaginary) radar directly. 
- The raw data frame is $N_s \times N_c \times N_{ant}$. $N_s$ is the number of samples, $N_c$ is number of chirps. After 2D FFT on this data frame, you get $N_D \times N_R \times N_{ch}$. Radar data is complex, thus $N_{ch} = 2N_{ant}$. 
- The study used radar calibration data, and thus need extensive data augmentation. Phase shift is used. --> This will essentially moves the object around in the range-doppler map. This gives more natural appearance to the RD map as opposed to doing conventional image augmentation (translation and mirroring, etc).
- The study only looks at one point per object (or center of the object), so segmentation is used as compared to object detection framework. 
- For direction of arrival, 3x3xCh local crop is sufficient for the Ang-Net.
- Metrics are measured by distances between gt and pred bins.
- Conventional method: RD accuracy~98%, Az accuracy ~90%, El accuracy ~73%.
 

#### Technical details
- Summary of technical details

#### Notes
- The phase shift augmentation part is puzzling. How does phase shift impact radar data? Would the annotation change as well?
- [Encyclopaedia Britanica](https://www.britannica.com/technology/radar/Factors-affecting-radar-performance) has a great summary article about radar performance.

