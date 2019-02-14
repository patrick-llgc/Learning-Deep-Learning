# [SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf)

_Feb 2019_

tl;dr: Understand video with two pathways, one slow pathway which understands the spatial information and one fast pathway which tracks the motion. This is **biologically inspired** by the P cells and M cells in retinal ganglion cells. 

#### Overall impression
The paper is quite eye-opening, in particular with the analogy to the [gangalion cells in the primate retinal system](http://thebrain.mcgill.ca/flash/a/a_02/a_02_cl/a_02_cl_vis/a_02_cl_vis.html). The P cells are sensitive to color with high acuity, but slow. The M cells are colorblind with low acuity, but fast. The M cells has a larger receptive field (Yes receptive field is a medical term). I believe this is the future direction of video recognition. 

#### Key ideas
- Spatiotemporal orientations are not equally likely.
  - The categorical spatial semantics of the visual system often evolve slowly. Therefore the recognition of the categorical semantics can be refreshed slowly.
  - But the motion being performed can evolve much faster then their subject identities. Therefore the potentially fast changing motion should be modeled using fast refreshing frames. 
- SlowFast networks has two pathways operating at two different temporal speed. A slow one operating at low frame rate to capture spatial semantics and a fast pathway operating at high frame rate to capture motion at fine temporal solution. **The fast pathway can capture motion without building a detailed spatial representation.** The fast pathway is lightweight but temporally high-resolution.
- Both pathways take input with the same spatial resolution. Slow and fast are synchronized. Slow means slow refreshing rate, not lagging behind.
  - In order to speed up processing in the fast pathway, the model used **fewer channels**, not lower spatial resolution. (Reduced spatial resolution may lose the ability to track spatial informaton, but reduced channel number makes the model more selective on what features related to motion to keep.)
  - Ablation test shows that gray scale input performs as well as RGB input. This is consistent with M cells being colorblind.
- SlowFast has multiple laternal connections. This is key to good performance. Multiple ways of connection (to resolve discrepancies in temporal dimension) all leads to good performance, but n_tx1x1 conv is the best. 

#### Technical details
- $\alpha$: The sampling rate of the two pathways has a ratio of $\alpha$ usually set to 8. That means the fast pathway refreshes 8 times as fast as the slow pathway.
- $\beta$: The fast pathway has a lower channel capacity. In primate visual systems only ~15 to 10% are M-cells. 
- $\tau$: The sampling rate of the flow path. $\tau / \alpha$ is the sampling rate of the fast path.
- In the slow pathway, temporal covolution is added only in later stages, as little correlation within a temporal receptive field unless the spatial receptive field is large enough. **[This is brilliant!]**
- Cosine schedule for training, with dropout only before classifier. 
- The fast pathway alone has 20% FLOPs, and does not perform well itself as it lacks capacity to learn spatial semantics. 
- Tradeoff of accuracy and FLOP by adjusting $\tau$. FLOPs can be reduced to 25% with 3% or so drop in accuracy.

#### Notes
- This is very similar to the coarse to fine approach to integrate global information to networks focusing on more local features. Maybe I should read more about the phisiology of primate visual systems to gain some intuition and new ideas. 
- A fancy word to denote the idea that the model is improved by introducing new architectures without increasing too much complexity/FLOPS in the backbone, is "orthoganal", as in "using advanced backbone is orthoganal to our new concept".

