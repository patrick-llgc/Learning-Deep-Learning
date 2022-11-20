# [M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction](https://arxiv.org/abs/2202.11884)

_November 2022_

tl;dr: Factor marginal prediction into conditional predictions, by classifying agent relationship into influencers and reactors. 

#### Overall impression
Trajectory prediction is used by autonomous driving cars to infer future motions of nearby agents and identify **risky scenarios** to enable safe driving. DL based methods excel at prediction marginal trajectories for single agents, but it remains an open problem to jointly predict **scene compliant** trajectories over multiple agents. 

Even [DesnseTNT](dense_tnt.md) cannot handle two agents, hundred goals for each agent.

#### Key ideas
- The system consists of relation predictor, marginal predictor, conditional predictor, sample selector. 
- Relation predictor: the influencer and the reactor
	- The predicted trajectory of the reactor is conditioned on the predicted trajectory of the influencer
- Pass/yield relation between two agents. 
	- TrafficGraphNet: pass, yield, none.
	- Compute the closest spatial distance between two agents. If the distance is smaller than a thresh, then there is no pass yield relationship.
	- If A gets to the interaction point first, then A is the influencer. Vice versa. 
- Marginal predictor and Conditional predictor
	- The marginal predictor is the same as the most single-agent marginal predictor.
	- Conditional predictor is largely the same takes in **influencer future trajectories**, as the augmented scene context input.
	- **Despite the prediction errors** of the conditioned agent, the model outperforms marginal predictors that do not consider interactive correlations.
- Sample selector
	- N trajectories for influencer
	- N reactor trajectories for each influencer trajectory
	- select K highest joint likelihood ones out of N^2 candidates. N=80 to ease computation.

#### Technical details
- **Metric OR** (overlap rate): measures the level of scene compliance as the percentage of the predicted traj of any agent overlapping with the predicted traj of other agents. The lower the better. --> This could also reflect the collision between a pred traj and a stationary vehicle (whose traj is one point in world coord).
- Encoder: vectorized (VectorNet) and rasterized (HOME), concatenated to get the best of both worlds. 
- The conditional predictor is trained with [**teacher forcing**](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/), with the GT. 
- The Relation predictor can reach an accuracy of 90%.
- Interesting Ablation (table 2): if we greedily select the top1 influencer traj, and the sequentially select the top1 reactor traj, then it is not as good as the naively combined marginal prediction. **We have to allow the topN (N=80) and use sample selector to make the M2I idea work.**
- The performance of M2I heavily depends on the size of interactive training data, especially when training the relation predictor and the conditional traj. --> Can we use smart agent simulation to generate training samples?

#### Notes
- Q: the relation predictor only selects the most likely relationship. --> Maybe extend this to multiclass and select the top K most likely relationship would be better. 
- Causality in driving interaction remains an open problem. The influencer and reactor are labelled by heuristics. 
- Due to uncertainty in human intent, the future trajectories are probabilistic and multimodal. 
- Prediction formulation
	- Mixture of Gaussians (Multipath, Trajectron++)
	- Generative models (GAN, VAE) suffer from sample inefficiency and require many samples to cover diverse driving scenarios. 
	- High level intentions
		- goal targets (TPNet, GOHOME, DenseTNT, PECNet, PRECOG, TNT)
		- lanes to follow (LaPred, PRIME)
		- maneuver actions (HYPER)
- Colliding trajectories
    - [End-to-end Contextual Perception and Prediction with Interaction Transformer](https://arxiv.org/abs/2008.05927) <kbd>IROS 2020</kbd> [Auxiliary collision loss, scene compliant pred]
    - [SafeCritic: Collision-Aware Trajectory Prediction](https://arxiv.org/abs/1910.06673) <kbd>BMVC 2019</kbd> [IRL, scene compliant pred]
- Interaction type prediciton
    - [TrafficGraphNet: Interaction-Based Trajectory Prediction Over a Hybrid Traffic Graph](https://arxiv.org/abs/2009.12916) <kbd>IROS 2020</kbd>
    - [Joint Interaction and Trajectory Prediction for Autonomous Driving using Graph Neural Networks](https://arxiv.org/abs/1912.07882) <kbd>NeurIPS 2019 workshop</kbd>
