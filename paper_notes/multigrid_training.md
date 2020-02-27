# [A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/abs/1912.00998)

_February 2020_

tl;dr: An efficient training technique by scaling spatial and temporal dimension of videos.

#### Overall impression
The paper is from FAIR and well written, as usual. Lots of experiments, and lots of GPUs (128)! Although they also validated the methods on 1 GPU as well with 3x speed up.

Recent Video training SOTA: [I3D](quo_vadis_i3d.md), [SlowFast](slowfast.md), [Non-Local](non_local_net.md)

It draws inspiration from [FixRes](fixres.md) that it requires a finetuning stage at the end to match train/test descrepancy.


#### Key ideas
- Use variable mini-batch shapes with different spatial-temporal resolutions that are varied according to schedule. Significant speedup for diff models, diff datasets, diff training settings (w/ or w/o pretraining, 128 vs 1 GPU).
- Train on coarse grid, then on finer grid, then finally finetune on same grid as inference setting. 
- Only require small changes to dataloader.

- Base Batch size BxTxHxW(x3) 
	- Long cycle
		- 8B x T/4 x H/sqrt(2) x W/sqrt(2)
		- 4B x T/2 x H/sqrt(2) x W/sqrt(2)
		- 2B x T/2 x H x W
		- B x T x H x W
	- Short cycle
		- Baseshape from long cycle
		- H/sqrt(2) x W/sqrt(2)
		- H/2 x W/2
- L-1 LR stage use multi-grid. Last LR stage use the baseline minibatch shape.
- Long cycle and short cycle
	- Mixture yields best performance
- Batch Normalization: standard batch size of 8 wrt long cycle. Increase batch size wrt short cycle.

#### Technical details
- Linear scaling rule
- Cosine learning schedule. This seems to yield similar performance to stagewise training schedule.
- Temporal subsampling: non-uniform stride
- May become I/O bound
- Training beyond 1 to 2 epoches hurt performance.

#### Notes
- Can we apply this to images?
- Temporal subsampling in the long cycle seem to hurt performance. Can we just downsample the spatial resolution? Short cycle do not downsample time and leads to better performance. Maybe the time dimension augmentation/subsampling altered the meaning of video.
