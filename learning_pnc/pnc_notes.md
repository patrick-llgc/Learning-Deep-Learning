# PnC Notes

## Introduction
- The notes were taken for the [Prediction, Decision and Planning for Autonomous driving](https://www.shenlanxueyuan.com/course/671) from Shenlan Xueyuan mooc course.
- The lecturer is [Wenchao Ding](website: https://wenchaoding.github.io/personal/index.html), former engineer at Huawei and not AP at Fudan University.

# Model-based Prediction
## Overview

- Model vs Learning
    - 8:2 —> 2:8, as sys evolves
    - Model-inspired leanring methods
- Planning of ego for next 8-10s. Need to do prediction.
- Prediction KPI
    - recall and precision on instance/event level, and their tradeoff.  误抢，误让
    - ADE and FDE not good metrics.
    - The tradeoff typically calls for a hybrid system model, learning vs model-based.
- Interaction: multimodality and uncertainty
    - Multimodality typically refers to multiple Intentions
    - Diff traj under same intention is not mentioned often.
    - Uncertainty rolls out, more uncertain into future.
- Model architecture
    - input representation, focus of 2016-2020
        - agents history
        - road graph + traffic light info
    - output representation, focus of 2021-now
        - intention/trajectory, multimodality, uncertainty
        - various types of agents
        - scenarios
    - Agent to map, agent to agent interaction.

# Constant velocity

- System evolution
    - 1st: CV and CT, baseline for fallback
    - 2nd: prediction based on manual features, prediction based on learning
    - 3rd: Long-horizon trajectory = intention + short-term trend
- CV
    - Pros: useful when mapless, lack of road info
    - Cons: windy road, agent-to-map falls back to CV, and CV can be very diff from actual map
    - tightly coupled with perception features (yaw, and velocity)
- CT (constant turn)
    - Extension of CV, typically CTCV, CT in short.
    - Yaw rate from perception or sensor fusion, can be noisy, and diff from actual map. Bad for U-turn and turns at intersection.
    - Naturally a noisy baseline with bad cases. How to digest these bad cases is a good question.

## Short-term vs long-term

- short-term: 2-3s, kinematics or network
- long-term: 8s+, intension is the key! Network gradually can take on long-term, but intension is still very important abstraction.

## Intention Prediction with manual crafted features

- Intention: classification based on predefined agent behaviors (LC, LT, RT, etc)
    - [Autonomous Driving Strategies at Intersections: Scenarios, State-of-the-Art, and Future Outlooks](https://arxiv.org/pdf/2106.13052) <kbd>ITSC 2021</kbd>
    - Cons: missing map info, lose intention modality.
- Why? Simplified system design, but can cover 90% of scenarios.
- All system consists of Input, model, and output.
- ML-based (SVM)
    - features: lateral distance to target lane, lateral velocity, dash/solid line, distance to front cars, etc. It can go up to tens of dimensions. Feature engineering is very important.
    - labeling: need to label “ahead of” LC. But by how much? —> This is the inherent issue of quantization of trajectory into classification buckets.
    - formulation: each SVM can predict whether to go to a lane or not for next timestep.
    - [Learning-Based Approach for Online Lane Change Intention Prediction](https://ieeexplore.ieee.org/document/6629564/) <kbd>IV 2013</kbd> [SVM, LC intention prediction]
    - Easy to implement, but hard to extend.
- DL-based
    - Simplified FE.
- Rule-based
    - Manually crafted SVM (or decision tree?)
- other output representation
    - Whether to go to a specific lane
    - Modeling intersection as 12 sectors
- Other input representation
    - Rasterize input as multichannel maps
    - pros: generalization capability
    - cons: lower efficiency, historical reasons (CNN, no transformers)
    - can be extended to BEV features
- Other models: HMM

## Model based Trajectory Prediction

- long-term trajectory prediction: lightweight planner
- Objective: min loss
    - confirm to short-term kinematics
    - conform to long-term intention
- Constraints
    - do not crash: map info, agent info
- Need to do super lightweight planner, 16, 32 or 64 agents. CPU-intensive.
    - typical planning methods (search, sampling, optimization) are two heavy
    - curve generation: fitting methods, such as bezier curves
- Example: Bezier traj generation
    - short-term: NN/CV/CT
    - long-term: long-term intention, and map query
    - Bezier to add control points to link short and long term intention planning, then de-duplicate controlling points.
- EPSILON
	- [EPSILON: An Efficient Planning System for Automated Vehicles in Highly Interactive Environments](https://arxiv.org/abs/2108.07993) <kbd>TRO 2021</kbd> [Wenchao Ding]

    - Intention prediction
        - Feature extraction: lane encoding, past traj of agent.
            - Lane encoder: needs to be adaptive to various num of lane line points
            - Coordinate transformation to relative coordinate systems of the targeted agent
            - Concatenation trick: duplicate trajectory encoding to all lanes
        - Output: n-way softmax vs n binary classifiers, softmax can be nasty to deal with for pnc as new information in one class will impact other classes.
    - Trajectory prediction: forward simulator, basically IDM (with acc), more friendly with interaction, more advanced format of traj geneartion.

    
# Path and Trajectory planning

- 路径轨迹规划三板斧：搜索，采样，优化. Typical planning methods, search, sampling, optimization)
- trajectory = path + speed

## Search

- Route/mission planning: 全局路径规划，巡进.
    - Find preferred route over road networks.
    - Graph search methods (Dijkstra, A*), DP in essence.
    - Input: start, end, lane graph, with cost function constraint
- Dijkstra's algorithm explores all possible paths to find the shortest one, making it a blind (uninformed) search algorithm (”天女散花”). It guarantees shortest path.
- A* algorithm is informed by using heuristics to prioritize paths that appear to be leading closer to the goal, making it more efficient (faster) by reducing the number of explored paths.
    - Uses a cost of cost so far (Dijkstra) + cost to go (heuristics, greedy best-first).
    - Only guarantees shortest path if the heuristic is admissible and consistent. If heuristics is bad, then it will make A-star worse than Dijkstra baseline, and will degenerate to greedy best-first.
    - [Very good visualization of A-star](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
    - A-star cons:  may not satisfy kinematics and cannot be tracked. E.g. steering angle typically 40deg.
- Hybrid A-star algo (A-star reflecting kinematics)
    - A-star maintains state and action both in grid space.
    - Why hybrid? It separates state and action.
    - Action: continuous space conforming with kinematics.
    - State (openset, cost): in discrete grid, more coarse.
- Extension of nodes in path in hybrid A-star
    - Need N discrete control action, e.g., discrete curvature.
    - Pruning: Multiple action history may lead to the same grid. Need pruning of action history to keep the lowest cost one.
    - Early stopping (analytical expansion, shot to goal): This is another key innovation in hybrid A-star. The analogy in A-star search algorithm is if we can connect the last popped node to the goal using a non-colliding straight line, we have found the solution. In hybrid A-star, the straight line is replaced by Dubins and RS (Reeds Shepp) curves. Maybe suboptimal but it focuses more on the feasibility on the further side. (近端要求最优性，远端要求可行性。)
- Dubins and RS curves
    - Dubins is essentially (arch-line-arch). RS curve improves Dubins curve by adding reverse motion.
- Holonomic vs non-holonomic (完整约束和非完整约束）
    - A holonomic constraint can be expressed as algebraic equations involving only the coordinates and time.
    - A holonomic constraint can be used to reduce one DoF of the system.
- Good review article https://medium.com/@junbs95/gentle-introduction-to-hybrid-a-star-9ce93c0d7869
- hybrid a-star is very useful in semi- or unstructured environment such as parking, or mapless driving.
- Define heuristics and cost needs heavy handcrafting.
- Cost = Progress cost g(n) + heuristics costs h(n)
    - Progress cost: length, steering change, conformation to centerline, penalty to use inverse lane and touching solid line, etc
    - heuristics is much more difficult. Two classical examples would be 1) kinematics constraints and neglecting obstacles and 2) obstacles and neglecting kinematics. —> Learning based heuristics?

## Sampling
- Safety and comfort
- How to guarantee consistency right from the design stage
- Why Frenet frame? Longitudinal vs lateral dynamics are very different.