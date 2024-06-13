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
    - Hybrid A-star: Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments <kbd>IJRR 2010</kbd> [Dolgov, Thrun, Searching]
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
- Safety and comfort, and how to guarantee consistency right from the design stage
- Frenet frame
    - Why? Longitudinal vs lateral dynamics are very different.
    - Decoupling based on reference line (RL).
    - SL coord system (SLT, or SDT): l_dot wrt t vs l_prime wrt s (curvature).
    - Pros: low/mid curvature.
    - Cons: Need to satisfy kinematics constraints in cartesian coord system. Hard to guarantee so when RL in extreme curvature.
- Transformation/projection Cartesian —> Frenet (SL coordinates)
    - Solve for projected point on RL, closest to CoM of the car.
    - Depends on the formulation of RL (For polyline, use brute-force iteration or binary search. For polynomial: optimization)
    - Need to pay attention to singularity.
    - [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.semanticscholar.org/paper/Optimal-trajectory-generation-for-dynamic-street-in-Werling-Ziegler/6bda8fc13bda8cffb3bb426a73ce5c12cc0a1760) <kbd>ICRA 2010</kbd> [Werling, Thrun]
- Consistency
    - Bellman's principle of optimality. In any optimal policy for a given problem, the remaining steps must also form an optimal policy for the remaining problem.
    - Replanned path should be temporally consistent. For every step in the planning, follow the remainder of the calculated trajectory, to provide temporal consistency.
    - OBVP (optimal bounded value problem), with initial state (position, v, a) and end state (pos, v, a), and min cost of integral of jerk squared. The solution to the optimal control problem is 5th order polynomial.
        - actually 5th polynomial (wrt time, not y wrt x!) is the optimal solution to a much more broader range of control problem.
    - Optimization by sampling. If we already know 5th polynomial is optimal, we can sample in this space and find the one with min cost to get the approximate solution.
    - Speed scenarios
        - High speed traj (5-6 m/s+): lateral s(t) and longitudinal d(t) can be treated as decoupled, and can be calculated independently.
        - Low speed traj: lateral and longitudinal are tightly coupled, and bounded by kinematics. If sampled independently, curvature may not be physically possible. Remedy: focus on s(t) and d(s(t)).
    - Eval and check after sampling
        - Selection based on min Cost
        - speed/acc/jerk limit, collision check.
    - Cons: simple road structure, short time horizon.

## Optimization
- Convex vs non-convex
    - convex: optimization (QP)
    - non-convex: DP searching or sampling, or nonlinear optimization (ILQR)
- QP: Apollo EM Planner
    - generate RL for each lane, perform optimization in parallel, and select optimal
    - decoupled path (space) and speed (temporal)
        - Cons: 窄道会车，空间轨迹会有交错，需要横纵联合优化。
    - [Baidu Apollo EM Motion Planner](https://arxiv.org/abs/1807.08048)
    - Optimization
        - SL projection (E)
        - Path Planning (M)
        - ST projection (E)
        - Speed planning (M)
    - In each M step, we have DP and QP step.
        - DP is to generate convex space
        - QP is to solve in convex space
    - M-step DP path: DP is essentially search, and it is necessary to solve non-convex problem (nudge, two local optima). If convex, then QP optimization should be sufficient.
        - Cost = smooth cost + obstacle（避障） + guidance (RL，贴线)
        - Obstacle avoidance typically introduces non-convex problems. If we know nudging direction, then it is an convex optimization problem (converting to lower and upper bound constraint). If we don’t know, then use DP to search.
    - M-step QP path: refine DP path
        - Linearize constraints by approxmization
        - the coarse solution from searching or sampling can only satisfy hard boundaries.
    - M-step DP speed optimizer
        - Cost = smooth cost + obstacle（避障, non-convex） + guidance (V_ref，交通规则)
        - DP converts to local optimization
        - forward-only search in ST graph (S and T is monotonous), as compared to SL 2D search
        - project predicted path of other agents into ego path SL, and into ST graph
    - M-step QP speed optimizer
        - refines DP speed as initial input
        - additional constraints such as jerk and acc limit
        - suppresses large acc and large jerk


# Joint spatiotemporal optimization

- lat/long decoupled: short temporal horizon.
- Previous basics can solve 95% of the case. Joint opt can solve the rest 5%.
- Why?
    - Decoupled solution in challenging dynamic interaction cases will lead to suboptimal trajectory.
    - Those 5% cases can show intelligence.
    - Current very hot topic.
- Methods
    - Search under spatiotemporal place
    - Iteration
    - spatiotemporal corridor
- Concrete example: Narrow space passing
    - optimal behavior: decel to yield or accel to pass
    - optimal behavior is not in the decoupled solution space, and need joint opt
- Challenges
    - high dimension, and non-convex —> time consuming
    - interaction —> even more complex, and denies brute-force methods. Not the focus of this course yet.

### Brute-force Search

- in xyt or slt
    - xyt: suitable for intersection, semi-structured (parking lot, or mapless) and unstructured (rural road).
    - slt: non-intersection, slt can save compute
- SLT space, 3D spatio-temporal map
    - long and flat. Like an energy bar. The visualization looks amazing!
    - projection to reference frame
    - SSC (state space corridor) in EPSILON
    - grid size: delta_t = 0.2 s
    - Expansion constraints in xot, yot and xoy plane. This leads to action space. Then we can use hybrid A-star.
    - Reference
        - [基于改进混合A*的智能汽车时空联合规划方法](https://www.qichegongcheng.com/CN/abstract/abstract1500.shtml)
        - [Enable Faster and Smoother
        Spatio-temporal Trajectory Planning for
        Autonomous Vehicles in Constrained
        Dynamic Environment](https://www.researchgate.net/publication/340516864_Enable_faster_and_smoother_spatio-temporal_trajectory_planning_for_autonomous_vehicles_in_constrained_dynamic_environment)
- Hybrid A star
    - Action space. Discretize steering and acceleration —> similar to tokenization
    - initial state (s, l, t, theta, v)
    - State update equations
    - Cost = g (progress cost) + h (cost to go, heuristics)
        - progress cost: collison, coherence with center line, etc.
        - heuristics
    - Issues with cost:
        - Can be 10+ items in a production systems.
        - The cost design will determine how “human-like” the trajectory will be.
    - search constraints: t time cannot reverse, and s cannot reverse.
    - how to optimize/recycle cost compute (eg., D-star)

## Iteration

- path → speed → path → speed (alternating minimization), like EM planner (E, M, E, M)
    - recognization of nudging point in next round
- improve brute-force search: can we focus on most probable part of the solution space? Get to the local opt first.
    - [Focused Trajectory Planning for Autonomous On-Road Driving](https://www.ri.cmu.edu/pub_files/2013/6/IV2013-Tianyu.pdf) <kbd>IV 2013</kbd>
    - Reference traj first (road shape, obstacle), tracking traj (kinematics and dynamics constraits, in perturbed neighborhood)
    - Reference traj (non-parametric)
        - Ref path: optimization of scattered points (v, a, kr are back calculated)
        - speed allocation, or speed optimization (highly nonlinear though)
    - (optional) linkage: reparameterize path with spirals
    - Tracking traj
        - sampling with 3rd polynomial, around reference traj
        - rate of traj based on cost wrt to ref traj
        - How much perturb is needed, is tradeoff
    - multitask extension: multiple ref trajectory, scale to multiple scenarios.

## Spatiotemporal semantic corridor (SSC)
- Elements of SSC
    - SLT space, static obstacle, dynamic obstacle
    - Abstraction: All semantic constraint can be converted to the constraint in SLT space.
    - SSC converts all semantic elements into constraint and converts traj generation into a optimization QP.
    - Reference
        - [SSC: Safe Trajectory Generation for Complex Urban Environments Using Spatio-Temporal Semantic Corridor](https://arxiv.org/abs/1906.09788) <kbd>RAL 2019</kbd> [Wenchao Ding]
        - [MPDM: Multipolicy Decision-Making for Autonomous Driving via Changepoint-based Behavior Prediction](https://www.roboticsproceedings.org/rss11/p43.pdf) <kbd>RSS 2011</kbd>
    - SSC is part of the [EPSILON codebase](https://github.com/HKUST-Aerial-Robotics/EPSILON)
- SSC generation Process
    - Seed generation, by projection of based on initial solution by BP (MPDM)
    - Cube inflation based on all seeds. This will create week links.
    - Cube relaxation to dilate week links.
- Generate trajectory based on QP optimization
    - parameterized optimization, use bazel and bernstein polynomials as basis. Adjust these points to generate bezel curve.
    - It is guaranteed that all curve is contained within the position convex hull extended by control points.
    - A piece of Bezel curve within each cube of the spatiotemporal corridor.
- For each cube of the corridor, then QP optimization
    - cost: jerk ** 2, very simple. It converts the complex cost design into semantic corridor generation.
    - continuous constraints for each piece.
- [Question] how was the narrow space passing solved in SSC? I feel SSC generation basically gets the DP done.



# Decision Making
## Markov Decision Process

- Why do we need decision making since we have planning already?
    - keywords: interaction and uncertainty. This makes the world probabilistic. Mainly from dynamic objects. —> **Interaction is actually the most difficult part of autonomous driving**, making it harder than robotics. It is like playing a game with probabilistic world model.
    - If the world is deterministic (geometry), there is no decision making needed. Searching, sampling, and optimization should be good enough.
    - For stochastic strategy, then MDP or POMDP.
    - Like RL, need to understand the entire problem setting to design S, A, R, and E.
    - To do e2e, we need to understand the decision making system first.
- Every decision is a bunch/cluster of planning.
- “Freezing robot”: prediction fills the entire S-T. 华山挤出一条路。Geometry —> Probability to the rescue.
- Markov process (MP)
    - stochastic process vs probability: dynamic random phenomenon, vs static
    - Markov only depend on current state, the current state is sufficient to predict the future.
    - For AD, maybe only last 1sec of state. Expand state space to trade for a shorter history window.
- Markov decision process (MDP)
    - can be modeled with 5 factors (State, Action, Reward, Probability of transfer, gamma)
    - extends a Markov Process to include decisions. It models decision-making process where outcomes are partly random and partly under the control of a decision maker.
    - It defines current state and action.
    - The state of other agents can be modeled as ego state or environment.
- Bellman’s equation
    - The best decision in any state is the one that maximizes your immediate reward plus the expected future rewards from the new state you reach.
    - $V(s) = \max_{a}[R(s,a) + \gamma \sum_{s'}P(s'|s,a)V(s')]$
- V and Q in RL
    - Value function (V) assesses the quality of states. Sum of expected return if you start with a state and play optimally.
    - Action-value function (Q) assesses the quality of actions within states.

## Value iteration and Policy iteration
- Value iteration
    - Value iteration is a method used in reinforcement learning to find the best strategy (or policy) by repeatedly updating the value of each state until it stabilizes.
    - How does this work if all states are initialized as zero? Immediate reward in Bellman’s equation is the key to brings in non-zero information. Value propagate throughout state space like a virus.
    - Value iteration is guarantee to converge for finite state. If gamma the discount factor = 0, then the value iteration will converge after 1 iteration. A smaller gamma leads to faster convergence as the consideration horizon is shorter.
    - Cons
        - ALL states needs to be considered!
        - ALL action needs to be evaluated: max a in the action space
        - Transition probability may contain uncertainty (slippage, etc).
    - Value iteration as the vanilla approach and gives the feasibility (like Dijkstra) but typically not useful in practice.
    - Reward is typically certain, but discount factor (far-sighted or short-sighted) and uncertainty would affect policy severely.
    - Value iteration to optimal policy: Use the optimal value function to determine the best action in each state, thereby obtaining the optimal policy. A policy spells out an action (or probability distribution) given a state, pi(s) = a.
    - $\pi^*(s) = \arg \max_a \left[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^*(s')\right]$
- Policy iteration (the improved version)
    - Value iteration requires to take the optimal action at each iteration. Policy iteration decouples policy evaluation and policy improvement. (每天进步一点点。)
    - Policy evaluation evaluates V^pi, basically V based on policy (without max a! This is faster). Any initial policy would do. This is an iterative step, but faster as each step is taken based on a given policy instead of exploring all possible actions (max a).
    - Policy improvement improves pi by generating pi’: greedily improve policy.
    - Pros of Policy Iteration
        - Can be more adaptable to changes in the environment. Once the initial policy evaluation step is done, the policy improvement step can quickly adjust to new data or changes in the urban environment.
        - Convergence speed: Practically policy iteration is faster than value iteration. Each iteration of policy iteration can be more computationally intensive than value iteration due to the policy evaluation step but fewer iterations are needed overall.
    
## AlphaGo

- Why MCTS? Why not value iteration and policy iteration?
    - Value iteration is a systematic, iterative method that updates the value of EVERY state based on expected future rewards (or a search tree that has unlimited breadth and width like Dijkstra), whereas MCTS dynamically builds and explores a subset of the search tree through simulations, focusing on promising future actions.
- MCTS vs value iteration
    - MCTS balances exploration and exploitation (via UCB definition), where value iteration focuses only on exploitation, in an exhaustive search way for a controlled known environment.
    - Value iteration is to MCTS as Dijkstra's algorithm is to (hybrid) A*: both value iteration and Dijkstra's systematically consider all possibilities (covering the entire search space) without heuristics, while MCTS and A* use heuristics to focus on the most promising options, making them more efficient for large and complex problems.
    - Value iteration aims to find an optimal policy for all states. MCTS focuses on decision-making from the current state. MCTS terminates based on practical constraints like time or iterations.
    - The fast rollout step in MCTS and the early termination (analytical expansion) in hybrid A* both serve to speed up their respective search processes by providing quick, approximate evaluations,
- MCTS: Tree search
    - Node: state
    - root node: current stage
    - leaf node: terminal or unexplored state
    - edge: action leading to another node
- MCTS: policy estimation, focuses on decision-making from the current state. It has 4 steps process of selection-expansion-simulation-backprop.
    - **Selection**: Follow the most promising path based on previous simulations until you reach a leaf node (a position that hasn’t been fully explored yet).
    - **Expansion**: add one or more child nodes to represent the possible next moves.
    - **Simulation**: From the new node, play out a random game until the end (this is called a “rollout”).
    - **Backpropagation**: Go back through the nodes on the path you took and update their values based on the result of the game. If you won, increase the value; if you lost, decrease it.
- MCTS guided by value network and policy network.
    - Value network reduce the search depth by summarizing values of sub-trees, so we can avoid going deep for good estimations. Policy network to prune search space. Balanced breadth and width.
    - MCTS used both value network and reward from rollout.
    - Policy network reduce the breadth of the search tree by identifying sensible moves, so we can avoid non-sensible moves.
    - Value network V evaluates winning rate from a state (棋面).
    - Trained with state-outcome pairs. Trained with much more self-play data to reduce overfit.
    - Policy network evaluates action distribution
    - Value network is more like instinct (heuristic), value network provides policy gradient to update policy network. Tesla learned collision network, and heuristic network for hybrid A-star.
- With autonomous driving
    - World model
    - AlphaGo tells us how to extract very good policy with a good world model (simulation)
    - Autonomous driving still needs a very good simulation to be able to leverage alphaGo algorithm. —> It this a dead end, vs FSD v12?
    - Tesla AI day 2021 and 2022 are heavily affected by AlphaGo. FSDv12 is a totally different ballgame though.
    - Go is limited and noiseless.
- Policy networks
    - P_s trained with SL, more like e2e
    - P_p trained with SL, shallower for fast rollout in MCTS
    - P_r trained with RL to play with P_s
- Training vs Inference
    - Training: PN + VN
    - Inference: MCTS + PN + VN
- Question: PN is essentially already e2e, why need VN and MCTS?
    - My guess: Small scale SL generate PN not strong enough, so need RL and MCTS to boost performance.
    - E2E demonstrates that with enough data, e2e SL can generate strong enough PN itself.
    - Maybe later MCTS will come back again to generate superhuman behavior for driving.
- Question: How to handle Prediction?
    - Conventionally people convert it to geometry (predicted waypoints)
    - A better way maybe treat it as the transition model and treat it in the decision making process. It can handle interaction quite well, but slow in compute.

![](https://pic4.zhimg.com/80/v2-d12f62f37b841e68e87802f80db861d7_1440w.webp)
- Alpha zero does not need SL to boostrap but purely relies on self-play to generate enough data.

## Safe RL
