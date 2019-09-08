# [TI white paper: Webinar: mmWave Radar for Automotive and Industrial applications
](https://training.ti.com/epd-pro-rap-mmwaveradar-adh-tr-webinar-eu)

_September 2019_

#### Key ideas
- Corner radars are SRR, for blind spot detection, lane change assist, etc
- SRR usually occupy 77-81 GHz, while LRR occupy 76-77 GHz
- Newer application: automated parking, in-cabin monitoring
- chirp: linear FMCW
- RF bandwidth vs IF bandwidth: RF bandwidth can be of few GHz, and IF bandwidth can be of few MHz. 
- Mixer multiplies Tx and Rx signal to get IF. 
- Roundtrip delay: td = 2R/c, beat frequency is B/Tc * td, chirp rate (frequency slope) and roundtrip delay.
- Doppler: doppler changes both the frequency and phase of the IF signal. However it is more sensitive to phase shift, and thus we focus on phase shift in FMCW radar. 
- Range-Doppler map: usually the detection of objects are done on this image. 
- Typical process flow: ADC --> preprocessing (DC removal, interf-mitigation) --> 1st FFT on range --> 2nd FFT on Doppler --> Detection (OS-CFAR, CA-CFAR) --> 3rd FFT Angle estimation --> raw point cloud --> clustering, tracking --> radar tracklets/pins
- Key design parameters:
	- Range resolution: dR = c/2B, range accuracy is small fraction of range resolution, and related to SNR
	- max velocity: v_max = lambda/4Tc. Tc is chirp duration. (intuitively, we want the sampling rate to be fast enough, among chirps)
	- velocity resolution: dv = lambda/2NTc (active duration of frame). velocity accuracy typically a fraction of the frame, 
	- angle resultion: dtheta = 2/K, at theta=0 and assuming d=lambda/2


#### Notes
- dbm: relative power to 1mWatt. 
- speed in m/s, such as 1 m/s is actually 2 mph. (x2.2 to be exact)
- advanced techniques is needed to extend max velocity
- longer window leads to better FFT results. radar DSP is all about recovering more information with limited side lobes with limited observation window. 