# [TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents](https://arxiv.org/pdf/1811.02146.pdf)

_June 2019_

tl;dr: Propose a 4D graph representation of trajectory prediction problem. The paper also introduced a new dataset in ApolloScape. 

#### Overall impression
The prediction seems not that good? Admittedly the dataset is focused on urban driving scenario and the problem is much harder than highway driving scenarios. The predicted trajectories projected on the single camera frame do not quite make sense either (some agents goes above the horizon and goes to the sky?).. 

#### Key ideas
- In the 4D graph (2D for spatial location and interaction of agents, one dim for time, and one dim for category), each instance is a node, and the relationships in spatial and temporal are represented by edges. Then each node and edge are modeled by a LSTM. 
- Two types of layers, instance layer and category layer. The main idea is to aggregate the average behavior of agents of a particular type, and then use it to finetune each individual agent's behavior.
- Three types of agents, bicyles, pedestrians and vehicles. 

#### Technical details
- Used LSTM with Self-attention from the tansformer paper.

#### Notes
- The simulation(NGSIM) dataset has trajectory data for cards but limited to highway with simlar simple road conditions. This could be useful for behavioral prediction and motion planning.

