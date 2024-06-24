# A Crash Course of Planning for Perception Engineers in Autonomous Driving 

🚧 Draft, for review only, 06/23/2024


> The fundamentals of planning and decision-making, a Medium blog post
> 

> By Patrick Langechuan Liu, 2024/06/23
> 


💡 Change log

- 06/19/2024: Wednesday, first draft with complete skeleton.
- 06/20/2024: Thursday, finish introduction.
- 06/21/2024: Friday, finish classical planning and industry practices.
- 06/22/2024: Saturday, finish decision making and industry practices.
- 06/23/2024: Sunday, finish reflection and format entire article.
- 06/24/2024: Monday, pass on to others to review.
- 06/25/2024: Tuesday, collect review feedbacks, format on Medium.
- 06/26/2024: Wednesday, publish on Medium.
- 06/27/2024: Thursday, translate to CN. 
- 06/28/2024: Friday, format on Zhihu. 
- 06/29/2024: Saturday, publish on Zhihu.

</aside>

![AlphaGo, ChatGPT and FSD (image credit [DeepMind](https://deepmind.google/technologies/alphago/), [Teslarati](https://www.teslarati.com/tesla-full-self-driving-12-rules-based/), and [Jonathan Kemper](https://unsplash.com/@jupp?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/a-computer-screen-with-a-text-description-on-it-5yuRImxKOcU?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash))](https://cdn-images-1.medium.com/max/1600/1*hCU8WQNRwjYqoR4jgE7GqA.png)

AlphaGo, ChatGPT and FSD (image credit [DeepMind](https://deepmind.google/technologies/alphago/), [Teslarati](https://www.teslarati.com/tesla-full-self-driving-12-rules-based/), and [Jonathan Kemper](https://unsplash.com/@jupp?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/a-computer-screen-with-a-text-description-on-it-5yuRImxKOcU?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash))

A classical modular autonomous driving systems typically consists of perception, prediction, planning and control (PnC). Until perhaps one year ago (2023), the contents of AI in most mass production autonomous driving systems enjoys the benefits of data-driven development in perception, and gradually tapers off towards downstream components. In stark contrast with the low penetration rate of AI in planning stack, end-to-end perception systems (BEV) have been deployed on mass production vehicles.

![Classical modular design of an autonomous driving stack, 2023 and prior (Chart created by author)](https://cdn-images-1.medium.com/max/1600/1*3_lBKtnK3oy1b7NSPiQ9cw.png)

Classical modular design of an autonomous driving stack, 2023 and prior (Chart created by author)

There are multiple reasons for this. A classical stack based on human crafted framework is more explainable and can be iterated faster to fix field test issues (on the order of hours) as compared to machine learning driven features (which could take up to days or weeks).  However in principle, it does not make sense to let cheaply available human driving data sitting idle. Additionally, increasing computing power is more scalable than expanding the engineering team.

Luckily there has been a strong trend in both academia and industry to change this situation. First of all, downstream modules will be increasingly more data-driven and may also be chained together via differentiable interface (in the style of CVPR 2023 best paper, UniAD). What is more, driven by the ever-burgeoning wave of Generative AI, a single unified vision-language-action (VLA) model shows great potential of handling complex robotics tasks (RT-2 in academia, TeslaBot and 1X in industry) or autonomous driving (GAIA-1, DriveVLM in academia, and Wayve AI driver, Tesla FSD in industry).

This brings the toolsets of AI and data-driven development from the stack perception to the stack of planning. 

This blog post aims to introduce the problem settings, existing methodology and challenges of the planning stack, in the form of a crash course to perception engineers. As a perception engineer I finally got some time to systematically learn the classical planning stack in the past couple of weeks and I would like to share what I learned. Of course I will also share my thoughts and what I believe how AI will help from the perspectives of an AI practitioner.

The intended audience of this post is AI practitioners who work in the field of autonomous driving, in particular perception engineers. 

# Why learn planning?

This brings us to an interesting question first. Why learn planning, especially the classical stack, in the era of AI?

From a problem-solving perspective, understanding your customers' challenges better will enable you as a perception engineer to serve your downstream customers more effectively, even if your main focus remains perception work.

Machine learning is a tool, not a solution. The most efficient way to problem solving is to combine new tools with domain knowledge, especially those with solid mathematical formulations. Domain knowledge-inspired learning methods are likely to be more data efficient. As planning transitions from rule-based to ML-based systems, even with early prototypes and products of end-to-end systems hitting the road, there is a need for engineers who can both deeply understand the fundamentals of planning and machine learning.  Notwithstanding these changes, it is probable that classical and learning methods will continue to coexist for a considerable period, perhaps shifting from an 8:2 to a 2:8 ratio. It would almost be a must for engineers working on this to understand both worlds.

From a value-driven development perspective, understanding the limitations of classical methods is crucial. This insight allows you to effectively utilize new ML tools to design a system that addresses current issues and delivers immediate impact.

Also, planning is a critical part of all autonomous agents, besides autonomous driving. Understanding what is planning and how planning works will enable more ML talents to work on this exciting topic to land truly autonomous agents, in the shape of cars or others. 

# What is planning ?

## The problem formulation

As the “brain” of autonomous vehicles, planning system is significant for the safe and efficient driving of vehicles. The goal of the planner is to generate trajectories that are safe, comfortable and progressing efficiently towards the goal. In other words, safety, comfort and efficiency are the three key goals for planning.

As input to the planning systems, all perception outputs are needed such as static road structures, dynamic road agents, free space generated by occupancy networks, and traffic wait conditions. They also need to ensure vehicle comfort by monitoring acceleration and jerk for smooth trajectories, while considering interaction and traffic courtesy.

The planning systems generates trajectories in a format of a sequence of waypoints for the ego vehicle’s low-level controller to track. Concretely, these are the positions of ego vehicle in the future at a series of fixed time stamps. For example, each point 0.4s apart to cover 8s planning horizon, totally 20 waypoints. 

A classical planning stack roughly consists of global route planning, local behavior planning, and local trajectory planning. Global route planning provides a road-level path from the start point to the end point on a global map. Local behavior planning decides on a semantic driving action type (e.g., car following, nudge, side pass, yield, and overtake) for the next several seconds. Based on the decided behavior type from behavior planning module, local trajectory planning generates a short-term trajectory. The global route planning is typically provide by map service once navigation is set, and is beyond the scope of this post. We will focus on the behavior planning and trajectory planning from now on. 

Behavior planning and trajectory generation can work explicitly in tandem or combined working as a whole. In explicit methods, behavior planning and trajectory generation are distinct processes operating within a hierarchical framework, and working at different frequencies with behavior planning at 1-5 Hz and trajectory planning at 10-20 Hz. Despite high efficiency most of the time, adapting to different scenarios may require significant modifications and finetuning. More advanced planning system combines the above two into a single optimization problem. This approach ensures feasibility and optimality without any compromise.

![Classification of planning design approaches (source: [Fluid Dynamics Planner](https://arxiv.org/abs/2406.05708))](https://cdn-images-1.medium.com/max/1600/1*fDiwxk5zMr0nKsn5bChWhQ.png)

Classification of planning design approaches (source: [Fluid Dynamics Planner](https://arxiv.org/abs/2406.05708))

## The Glossary of Planning

You might have noticed that the terminology used in the above session and the image does not completely match. There is no standard terminology that every body uses. Across both academia and industry, it is not uncommon for engineers to use different names to refer to the same thing and the same name to refer to different things. This is indicative that planning in autonomous driving is still under active development and has not really converged.

Here I list out the notation used in this post, and also explain briefly what other notions present in literature.

- Planning: top level concept, in peers to control, and generates trajectory waypoints. Together they are jointly referred to as PnC (planning and control).
- Control: top level concept, takes in trajectory waypoints and generates high frequency steering, throttle and brake commands for actuators to execute. Control is pretty much a solved problem relative to others and is beyond the scope of this post, despite the common notion of PnC.
- Prediction: top level concept, predicts the future trajectories of traffic agents other than ego. Prediction can be considered as a lightweight planner for other agents. Also called motion prediction.
- Behavior planning: module which produces high-level semantic actions (lane change, overtake, etc) and typically produces a coarse trajectory. Also called task planning. Or decision making, in particular in the context of interaction.
- Motion planning: module which takes in semantic actions and produces smooth, feasible trajectory waypoints for the duration of the planning horizon for control to execute.
- Trajectory planning: same as motion planning.
- Decision making: behavior planning with a focus on interactions. Without ego-agent interaction, we will just say behavior planning. Also as tactical decision making.
- Route planning: Finds preferred route over road networks. Same as mission planning.
- Model-based approach: in planning, this model is NOT neural network model, but rather manually crafted frameworks used in classical planning stack. Model-based methods contrasts with learning based methods.
- Multimodality: in the context of planning, multimodality typically refers to multiple intentions. This contrasts with multimodality in the context of the multimodality sensors input to perception or multimodality large language models (such as VLM or VLA).
- Reference line: a local (several hundred meters) and coarse path based on global routing information and current state of ego vehicle.
- Frenet coordinates: a coordinate system based on a reference line. Frenet simplifies the a curvy path in Cartesian to a straight tunnel model. See below for a more detailed introduction.
- Trajectory: a 3D spatiotemporal curve, in the form of (x, y, t) in Cartesian coordinates or (s, l, t) in Frenet coordinates. Trajectory = path + speed.
- Path: a 2D spatial curve, in the form of (x, y) in Cartesian coordinates or (s, l) in Frenet coordinates.
- Semantic action: A high level abstraction of action (e.g., car following, nudge, side pass, yield, and overtake) with clear human intention. Also as intention, policy, maneuver, primitive motion.
- Action: there is no fixed meaning of this. From the finest degree, this can refer to the output of control (high frequency steering, throttle and brake commands for actuators to execute) or the output of planning (trajectory waypoints). Semantic action refers to the output of behavior prediction.

Different literature may have different notations of concept.

- Sometimes decision making system includes planning and control as well (source: [A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles](https://arxiv.org/pdf/1604.07446), and BEVGPT)
- Sometimes motion planning is the top level  planning concept and it includes behavior planning and trajectory planning (source: [Towards A General-Purpose Motion Planning for Autonomous Vehicles Using Fluid Dynamics](https://arxiv.org/abs/2406.05708)).
- Sometimes planning includes behavior planning and motion planning, and also includes route planning as well.

## Behavior Planning

As a machine learning engineer you may notice that the behavior planning module is a heavily manually crafted intermediate module. There is no consensus on the exact form and content of the output. Concretely, the output of BP can be a reference path, or objects labeling on ego maneuver (pass from left or right hand side, pass or yield, etc). The semantic action has no strict definition, no fixed methods.

The decoupling of behavior planning and motion planning is for increased efficiency in solving in the extremely high dimension action space of autonomous vehicles. The actions of an autonomous vehicle have to be reasoned at typically 10 Hz or more (time resolution in waypoints) and most of the actions of an autonomous vehicle is relatively boring going straight. After decoupling, the behavior planning layer only needs to reason about the future scenario at a relatively coarse resolution, while the motion planning layer operates in the local solution space given the decision. Another benefit of BP is on converting non-convex optimization to convex optimization, which we will see below soon.

## Frenet vs Cartesian systems

The Frenet coordinate system is a widely adopted coordinate system that merits its own introduction section. The Frenet frame simplifies trajectory planning by independently managing lateral and longitudinal movements relative to a reference path. The s coordinate represents longitudinal displacement (distance along the road), while the l (or d) coordinate represents lateral displacement (side position relative to the reference path).

Frenet simplifies the a curvy path in Cartesian to a straight tunnel model. This transformation converts non-linear road boundary constraints on curvy road into linear ones, significantly simplifying the subsequent optimization problems (as shown in the chart below). In addition, humans perceive longitudinal and lateral movements differently, and the Frenet frame allows for separate and more flexible optimization of these movements.  

![ Schematics on the conversion from Cartesian frame to Frenet frame (source: [Cartesian Planner](https://ieeexplore.ieee.org/document/9703250))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/102cec79-ad8b-468d-b4d7-8611701812cb/Untitled.png)

 Schematics on the conversion from Cartesian frame to Frenet frame (source: [Cartesian Planner](https://ieeexplore.ieee.org/document/9703250))

Frenet needs clean, structured road graph with low curvature lanes. In practice, for structured roads with small curvature (e.g., highways or city expressways), the Frenet coordinate system is preferred. The issues with the Frenet coordinate system are amplified with increasing reference line curvature, so it should be used cautiously on structured roads with high curvature (e.g., city intersections with guide lines). Therefor for unstructured roads, such as ports, mining areas, parking lots, or intersections without guide lines, the more flexible Cartesian coordinate system is recommended.

# Classical tools — the troika of planning

Planning in autonomous driving involves computing a trajectory from an initial high-dimensional state (including position, time, velocity, acceleration, and jerk) to a target subspace, ensuring all constraints are satisfied. Searching, sampling and optimization are the three most widely used tools for planning. 

## Searching

Classical graph-search methods are popular in planning, and they are used in route/mission planning on structured road, or directly on motion planning to find the best path in unstructured environment (such as parking, or urban intersection, especially mapless).

There is a clear evolution path, from Dijkstra, A-star (A*) and to hybrid A-star.

Dijkstra's algorithm explores all possible paths to find the shortest one, making it a blind (uninformed) search algorithm. It is a systematic methods to guarantees optimal path, but it is inefficient to deploy. In the chart below we can see that it explores almost all directions. And yes Dijkstra’s algorithm is essentially a breadth-first search (BFS) weighted by movement costs, and obviously we can use information of the location of the target to trim down the search space. 

![Visualization of Dijkstra’s algorithm and A-star search (Source: [PathFinding.js](https://qiao.github.io/PathFinding.js/visual/), example inspired by [RedBlobGames](https://www.redblobgames.com/pathfinding/a-star/introduction.html))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/ec37a915-831a-4355-b820-2b9bb5e79a3d/Untitled.png)

Visualization of Dijkstra’s algorithm and A-star search (Source: [PathFinding.js](https://qiao.github.io/PathFinding.js/visual/), example inspired by [RedBlobGames](https://www.redblobgames.com/pathfinding/a-star/introduction.html))

A* algorithm does exactly that by using heuristics to prioritize paths that appear to be leading closer to the goal, making it more efficient. It uses a combination of cost so far (Dijkstra) + cost to go (heuristics, essentially greedy best-first). Only guarantees shortest path if the heuristic is admissible and consistent. If heuristics is bad, then it will make A-star worse than Dijkstra baseline, and will degenerate to greedy best-first.

In the specific application of autonomous driving, hybrid A-star algorithm further improves A-star by considering vehicle kinematics. A-star may not satisfy kinematics and cannot be tracked (e.g., steering angle typically is within 40 degrees). While A-star operates in grid space for both state and action, hybrid A-star separates them, maintaining state in grid but allowing continuous action per kinematics.

Analytical expansion (shot to goal) is another key innovation proposed by hybrid A-star. By analogy, a natural enhancement to A-star is to connect the most recently explored nodes to the goal using a non-colliding straight line. If this is possible, we have found the solution. In hybrid A-star, this straight line is replaced by Dubins and RS (Reeds Shepp) curves, which comply with vehicle kinematics. This early stopping methods strikes a balance between optimality and feasibility, by focusing more on the feasibility on the further side. 

Hybrid A-star used heavily in parking scenario and mapless urban intersections. Here is a very nice video showcasing how it is working in a parking scenario.

![Hybrid A-star algorithm with analytical expansion (source: the [2010 IJRR Hybrid A-star paper](https://www.semanticscholar.org/paper/Path-Planning-for-Autonomous-Vehicles-in-Unknown-Dolgov-Thrun/0e8c927d9c2c46b87816a0f8b7b8b17ed1263e9c) and [2012 Udacity class](https://www.youtube.com/watch?v=qXZt-B7iUyw&ab_channel=Udacity) )](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/3f711448-b344-4e8b-936f-596f034f7162/Untitled.png)

Hybrid A-star algorithm with analytical expansion (source: the [2010 IJRR Hybrid A-star paper](https://www.semanticscholar.org/paper/Path-Planning-for-Autonomous-Vehicles-in-Unknown-Dolgov-Thrun/0e8c927d9c2c46b87816a0f8b7b8b17ed1263e9c) and [2012 Udacity class](https://www.youtube.com/watch?v=qXZt-B7iUyw&ab_channel=Udacity) )

## Sampling

Another popular method of planning is sampling. The well-known Monte Carlo method is a random sampling method. For sampling-based methods, fast evaluation of many options is critical as it directly impacts real-time performance of the autonomous driving system.

> LLM is essentially providing samples, and there needs to be an evaluator with some cost, aligned with human preference.
> 

Sampling can happen in parameterized solution space if we already know the analytical solution to a given problem or subproblem. For example, typically we want to minimize the time integral of the square of jerk (third derivative of position p(t), thus the triple dot on top of p, with one dot meaning one order derivative to time), among many others. 

![Minimizing squared jerk for driving comfort (source: [Werling et al](https://www.semanticscholar.org/paper/Optimal-trajectory-generation-for-dynamic-street-in-Werling-Ziegler/6bda8fc13bda8cffb3bb426a73ce5c12cc0a1760), ICRA 2010)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/b2141370-c133-459e-bcfd-0f47dc2b3ec8/Untitled.png)

Minimizing squared jerk for driving comfort (source: [Werling et al](https://www.semanticscholar.org/paper/Optimal-trajectory-generation-for-dynamic-street-in-Werling-Ziegler/6bda8fc13bda8cffb3bb426a73ce5c12cc0a1760), ICRA 2010)

It can be mathematically proven that the quintic (5th order) polynomials are the jerk-optimal connection between two states in a position-velocity-acceleration space, even if there are other additional cost terms. Then we can sample in this parameter space of quintic polynomial and find the one with min cost to get the approximate solution. The cost will take into account the speed, acceleration, jerk limit, and collision check. This way, we are essentially solving the optimization problem by sampling.

![Sampling of lateral movement time profiles (source: [Werling et al](https://www.semanticscholar.org/paper/Optimal-trajectory-generation-for-dynamic-street-in-Werling-Ziegler/6bda8fc13bda8cffb3bb426a73ce5c12cc0a1760), ICRA 2010)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/8ea5d572-d3a0-4bbd-807f-6689b62e96f2/Untitled.png)

Sampling of lateral movement time profiles (source: [Werling et al](https://www.semanticscholar.org/paper/Optimal-trajectory-generation-for-dynamic-street-in-Werling-Ziegler/6bda8fc13bda8cffb3bb426a73ce5c12cc0a1760), ICRA 2010)

Sampling-based methods have inspired numerous ML papers, including CoverNet, Lift-Splat-Shot, NMP, and MP3. These methods replace mathematically sound quintic polynomials with human driving behavior, utilizing a large database. The evaluation of trajectories can be easily parallelized, which further supports the use of sampling-based methods. This approach effectively leverages a vast amount of expert demonstrations to mimic human-like driving behavior, while avoiding random sampling of acceleration and steering profiles.

![Sampling from human driving datasets for data-driven planning methods (source: NMP, CoverNet and Lift-splat-shoot)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/984a266b-a2d7-4e22-9169-9838f29cbd11/Untitled.png)

Sampling from human driving datasets for data-driven planning methods (source: NMP, CoverNet and Lift-splat-shoot)

## Optimization

Optimization finds the best solution to a problem by maximizing or minimizing a specific objective function under given constraints. In neural network training, a similar principle is followed using gradient descent and backpropagation to adjust the network's weights. However, in optimization tasks outside of neural networks, models are usually less complex, and more effective methods than gradient descent are often employed. (For example, while gradient descent can be applied to Quadratic Programming, it is generally not the most efficient method.)

For autonomous driving, the planning cost to optimize typically considers the the dynamic objects for obstacle avoidance, static road structures for following lanes, navigation information to make sure the planning, and ego status to evaluate smoothness.

Optimization can be categorized into convex and non-convex types. The key distinction is that in a convex optimization scenario, there is only one global optimum and also the local optimum. This characteristic makes it unaffected by the initial solution to the optimization problems. For non-convex optimization, initial solution matters a lot, as illustrated in the chart below.

![Convex vs non-convex optimization (source: [Stanford course materials](https://stanford.edu/~pilanci/papers/TALK_Sketching.pdf))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/1ae6b730-24dd-4ebb-9833-28dd40bd5730/Untitled.png)

Convex vs non-convex optimization (source: [Stanford course materials](https://stanford.edu/~pilanci/papers/TALK_Sketching.pdf))

As planning is highly non-convex optimization and with many local optimum. This heavily depends on Initial solution. In addition, convex optimization typically runs much faster and thus are preferred for onboard realtime applications such as autonomous driving. A typically solution is that convex optimization is used with other methods to outline a convex solution space first. This is exactly the math foundation behind the idea of separating behavior planning and motion planning, and finding a good initial solution is actually behavior planning. 

Take obstacle avoidance as an concrete example, which typically introduces non-convex problems. If we know nudging direction, then it is an convex optimization problem, and the obstacle position act as lower or upper bound constraint for the optimization problem. If we don’t know, then we need to make a decision first which direction to nudge from, to make the problem a convex problem for motion planning to solve. The nudging direction decision is behavior planning.

> Of course we can do direct optimization of the non-convex optimization problem with tools such as Iterative Linear Quadratic Regulator (ILQR), but this is beyond the scope of the post.
> 

![A convex path planning problem vs a non-convex one (chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/7472c657-43d4-4a04-b79e-ed1fda18d321/Untitled.png)

A convex path planning problem vs a non-convex one (chart made by author)

![The solution process of the convex vs non-convex path planning problem (chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/65dbc9e9-7e6a-4522-964f-c64b41eca923/Untitled.png)

The solution process of the convex vs non-convex path planning problem (chart made by author)

How do we make such decisions? We can use the above mentioned search or sampling methods to address non-convex problems. Sampling-based methods scatter many options across the parameter space, effectively handling non-convex issues in a manner similar to searching.

You may also have the question why deciding which direction to nudge from is good enough to guarantee the problem space is convex? We need to talk more about topology. In path space, similar feasible paths can transform continuously into each other without obstacle interference. These similar paths, grouped as "homotopy classes" in the formal language of topology, can all be explored using a single initial solution homotopic to them. All these paths form a driving corridor, the red or green shaded area in the image above. For a 3D spatiotemporal case, please refer to this [QCraft tech blog](https://zhuanlan.zhihu.com/p/551381336).

> We can use [Generalized Voronoi diagram](https://zhuanlan.zhihu.com/p/551381336) to enumerate all homotopy classes, this roughly corresponds to how many decisions we can make. Yet this is beyond the scope of the blog post.
> 

The key to the efficiency of optimization problem solving lies in the capability of the optimization solver. Typically a solver requires on the order of 10ms to plan a trajectory. If we can boost this efficiency by 10x, this can significantly impact the algorithm design. This is exactly what happened to Tesla, as revealed in Tesla AI day 2022. This is very similar to what happens in perception, with 2D perception to BEV when available compute scaled up 10x. 

With a more efficient optimizer, more options can be calculated and then evaluated. In this sense, the importance of decision making can be reduced. The engineering of an efficient optimization solver requires tons of engineering resources.

> Every time compute scales up by 10x, algorithm will evolve to next generation.——The unverified law of algorithm evolution
> 

# Industry practices of planning

The key differentiator of various planning system is that whether it is spatiotemporally decoupled. Concretely, spatiotemporally decoupled methods plans in spaces first to generate a path, and than plan the speed profile along the path. Thus spatiotemporally decoupled approach is also referred to path-speed decoupling. 

Path-speed decoupling is also referred to as lat-long decoupling, where lateral (lat) planning roughly equals to path planning and longitudinal (long) planning roughly equals to speed planning. This naming seems to be a legacy from the Frenet coordinate, which we will take a look later.

Decoupled solution is easier to implement, and can solve 95% of the issues. Coupled solution has higher performance ceiling in theory, but is more challenging in engineering details. More parameters to tune, and need more principled way to tune. 

![The comparison of decoupled and joint planning (source: [Qcraft](https://www.xchuxing.com/article/63767))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/f3aa976c-81bf-4946-8b41-671276850056/Untitled.png)

The comparison of decoupled and joint planning (source: [Qcraft](https://www.xchuxing.com/article/63767))

| Decoupled Spatiotemporal Algorithms | Joint Spatiotemporal Planning |
| --- | --- |
| Calculates solutions for two-dimensional spaces in two steps, losing the solution space for one dimension at each step | Directly calculates the optimal solution in three-dimensional space, increasing the solution space by one dimension |
| Faster solution speed, lower computational requirements | Fully considers dynamic obstacle information, making path planning more reasonable |
| May fall into suboptimal trajectory problems in complex dynamic scenarios | Matches human driving habits, suitable for directly learning human driver behavior |

## Path-speed decoupled planning

We can take [Baidu Apollo EM planner](https://arxiv.org/abs/1807.08048) as one example that uses path-speed decoupled planning.

EM planner significantly reduces computational complexity by transforming a three-dimensional station-lateral-speed problem into two two-dimensional station-lateral/station-speed problems.

At the core of Apollo EM planner is an iterative EM step, consisting of path optimization and speed optimization, each dividing into a E-step (projection, formulation in 2D state space) and a M-step (optimization in the 2D state space). The E-step involves projecting the 3D problem into either a Frenet SL frame or ST speed tracking frame.

![The EM iteration in Apollo EM planner (source: [Baidu Apollo EM planner](https://arxiv.org/abs/1807.08048) )](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/d1864a49-faf5-44f1-ab99-571b37b91888/Untitled.png)

The EM iteration in Apollo EM planner (source: [Baidu Apollo EM planner](https://arxiv.org/abs/1807.08048) )

The M optimization step for both path and for speed are non-convex optimization problems (For path optimization, whether to nudge an object on the left or right hand side of it. For speed optimization, whether to overtake or yield to a dynamic object that cross its path with ours). Apollo EM planner tackles the non-convex optimization issue by a two step DP-then-QP process. DP (dynamic programming) uses sampling or searching algorithm to generate a rough initial solution and prune the non-convex space to a convex space. QP (quadratic programming) takes coarse DP results as input and optimizes in the convex space provided by DP. In other words, DP focuses on feasibility and QP focuses more on optimality by refining initial solution in the convex space.

In our defined terminology, Path DP corresponds to lateral BP, Path QP to lateral MP, Speed DP to longitudinal BP, and Speed QP to longitudinal MP. It is conducting BP then MP in both the path and the speed step.

![A full autonomous driving stack with path-speed decoupled planning (chart made by author)](https://cdn-images-1.medium.com/max/1600/1*UBfD_OBlts1rAX_55lobww.png)

A full autonomous driving stack with path-speed decoupled planning (chart made by author)

## Joint spatiotemporal planning

Although decoupled planning can resolve 95% of the cases in autonomous driving, for the rest of the 5%, a decoupled solution in challenging dynamic interactions often results in suboptimal trajectories. In the 5% of cases where complex interactions occur, demonstrating intelligence is crucial, making this a very hot topic.

For example, in narrow space passing, the optimal behavior might be to decelerate to yield or accelerate to pass. Such optimal behaviors are not achievable within the decoupled solution space and require joint optimization.

![A full autonomous driving stack with joint spatiotemporal planning (chart made by author)](https://cdn-images-1.medium.com/max/1600/1*hqhJyBdq61ZMIENawTjBOQ.png)

A full autonomous driving stack with joint spatiotemporal planning (chart made by author)

However there are challenges in joint spatiotemporal planning. First of all, solving the non-convex problem directly in a higher dimension state space is more challenging and time consuming than the decoupled solution. Secondly, considering interaction in spatiotemporal joint planning is even more challenging. We will cover this topic later when we talk about decision making.

Here we introduce two solving methods, either brute force search, or constructing a spatiotemporal corridor for optimization.

Brute force search occurs directly in 3D spatiotemporal space (2D in space and 1D in time), and can be performed in either XYT (Cartesian) or SLT (Frenet) coordinates. We will take SLT for example. SLT space is long and flat. Like an energy bar. It is long in L dimension, flat in the ST face. For brute force search, we can use hybrid A-star, with cost a combination of progress cost and cost to go. During optimization, we have to conform to search constraints that we cannot reverse in both s and t dimension.

![Overtake by lane change in spatiotemporal lattice (source: [Spatiotemporal optimization with A*](https://www.qichegongcheng.com/CN/abstract/abstract1500.shtml))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/5ed18c23-17a7-4ee6-9486-09ee4d6327f7/Untitled.png)

Overtake by lane change in spatiotemporal lattice (source: [Spatiotemporal optimization with A*](https://www.qichegongcheng.com/CN/abstract/abstract1500.shtml))

Another method is constructing a spatiotemporal corridor. [SSC (spatiotemporal semantic corridor, RAL 2019)](https://arxiv.org/abs/1906.09788) encodes the requirements given by the semantic elements into a semantic corridor and a safe trajectory is generated accordingly. The semantic corridor consists of a series of mutually connected collision-free cubes with dynamical constraints posed by the semantic elements in the spatiotemporal domain. In each cube it is a convex optimization problem that can be solved by QP. 

SSC still needs a BP module to provide a coarse driving trajectory. Complex semantic elements of the environment are projected to the spatiotemporal domain w.r.t. the reference lane. [EPSILON (TRO 2021)](https://arxiv.org/abs/2108.07993) demonstrates a system with SSC as the motion planner working in tandem with a behavior planner. In the next session we will talk more about the behavior planning, especially with a focus on interaction. In this context, behavior planning is usually referred to as decision making.

![An illustration of the spatiotemporal corridor (source: [SSC](https://arxiv.org/abs/1906.09788))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/9b8615a4-68c3-4027-984c-062afe1db563/Untitled.png)

An illustration of the spatiotemporal corridor (source: [SSC](https://arxiv.org/abs/1906.09788))

# Decision making

## What and why?

Decision making is essentially behavior planning, but with a focus on interaction with other traffic agents. The assumption is that all other agents are mostly rational and will act to our behavior most of the time. We can also name this “noisily rational”.

People may question the necessity of decision making given that we have planning already and the powerful tools we can leverage. However, two key aspects—uncertainty and interaction—make the world probabilistic, primarily due to dynamic objects. Interaction is the most challenging part of autonomous driving, distinguishing it from general robotics.

In a deterministic (purely geometric) world without interaction, decision making would be unnecessary, and planning through searching, sampling, and optimization would suffice. Brute force searching in the 3D XYT space could serve as a general solution. 

In most classical autonomous driving stack, a prediction-then-plan approach is adopted, assumes zero-order interaction between ego and other vehicles. This treats prediction output as deterministic and ego has to react to them accordingly. This leads to overly conservative behavior, exemplified by the "freezing robot" problem. In such cases, prediction fills the entire spatiotemporal space, preventing actions like lane changes in crowded conditions—something humans manage more effectively.

For handling stochastic strategies, MDP or POMDP frameworks are essential. These approaches shift the focus from geometry to probability, helping to address chaotic uncertainty. By assuming that traffic agents behave rationally or at least noisily rationally, decision making can help create a safe driving corridor in the otherwise chaotic spatiotemporal space.

Among the three overarching goals of planning—safety, comfort, and efficiency—decision making primarily enhances efficiency. Conservative actions can maximize safety and comfort, but effective negotiation with other road agents, achievable through decision making, is essential for optimal efficiency. Effective decision making also displays intelligence. 

## MDP and POMDP

We will first introduce MDP and POMDP, followed by their systematic solutions, such as value iteration and policy iteration.

A **Markov Process (MP)** is a type of stochastic process, dealing with dynamic random phenomena, unlike static probability. In a Markov Process, the future state depends only on the current state, making it sufficient for prediction. For autonomous driving, the relevant state may only include the last second of data, expanding the state space to allow for a shorter history window.

A **Markov Decision Process (MDP)** extends a Markov Process to include decision-making by introducing action. MDPs model decision-making where outcomes are partly random and partly controlled by the decision maker or agent. It can be modeled with five factors: State (of the environment), Action (the agent can take to affect the environment), Reward (the environment can provide for the agent due to the action), Probability of Transfer (of the environment from the old state to new state upon agent’s action), and Gamma (a discount factor toward future reward). This is also the common framework used by reinforcement learning (RL), which is also an MDP. 

The goal of MDP or RL is to maximize the cumulative reward it receives in the long run. This requires the agent to make good decisions given a state from the environment, according to a policy. A policy, π, is a mapping from each state, s ∈ S, and action, a ∈ A(s), to the probability π(a|s) of taking action a when in state s. MDP or RL studies the problem of how to get the optimal policy. 

![The agent-environment interface in MDP and RL (source: [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/1a2e436f-1bfa-4785-83d7-b4b9cbdc3a2d/Untitled.png)

The agent-environment interface in MDP and RL (source: [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf))

A **Partially Observable Markov Decision Process (POMDP)** adds one extra layer of complexity that the states cannot be obtained directly but rather measured through observation. A belief is maintained within the agent to estimate of state of the environment. Autonomous driving scenarios are better represented by POMDPs due to their inherent uncertainties. MDP is a special kind of POMDP where observation is state.

![MDP vs POMDP (source: [POMDPs as stochastic contingent planning](https://www.researchgate.net/figure/MDP-and-POMDP-visualization_fig1_374986767))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/dc29b683-2e8d-42ef-b6d0-dc00aa4e2096/Untitled.png)

MDP vs POMDP (source: [POMDPs as stochastic contingent planning](https://www.researchgate.net/figure/MDP-and-POMDP-visualization_fig1_374986767))

POMDPs can actively collect information, leading to actions that gather necessary data, demonstrating the intelligent behavior of these models. It can really shines in scenarios like waiting at intersections.

## Value iteration and Policy iteration

**Value iteration** and **policy iteration** are systematical methods to solve MDP or POMDP problems. Although they are not used in reality, it would be good to have some understanding of the exact solution and how we can simplify it in real life, such as MCTS in AlphaGo or MPDM in autonomous driving.

From MDP, in order to find the best policy, we have to assess the potential or expected reward we can get from a state, or more concretely with an action from that state. (Note that this is not necessarily the immediate reward, but potentially all future rewards. More formally this is called a return, cumulative discounted reward. This is beyond the topic of this post, and please refer to the bible of RL: [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) for more details.)

Value function (V) characterizes the quality of states. It is the sum of expected return. Action-value function (Q) assesses the quality of actions for a given state. Both value functions and action-value functions are defined according to a given policy. Bellman Optimality Equation of optimality states that states that, an *optimal* policy would pick the action that *maximizes* the immediate reward plus the expected future rewards from the reachable new states.  In plain language, Bellman Optimality Equation tells us do not focus solely on the immediate reward but also consider where you will be as the consequence of the action and your exit options. For example, when you switch jobs, do not focus only on the immediate pay raise you can get (R), but also consider the value (S’) of this new position will offer you in the future.

![Bellman’s equation of optimality (chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/f24676ff-15c2-4631-b9c5-6bd5c8eee1fb/Untitled.png)

Bellman’s equation of optimality (chart made by author)

It is relatively straightforward to extract the optimal policy from Bellman Optimality Equation with the availability of the optimal value function. Then how do we find the optimal value iteration? Value iteration comes to rescue.

![Extract best policy from optimal values (chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/6920e226-bb2e-4474-b20f-88462665aabb/Untitled.png)

Extract best policy from optimal values (chart made by author)

Value iteration finds the best policy by repeatedly updating the value of each state until it stabilizes. Value iteration is obtained simply by turning the Bellman Optimality Equation into an update rule. Essentially we are using the optimal future picture to guide iteration toward it. In plain language, fake it until you make it!

![Update value functions under the guidance of Bellman’s Equation (chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/add160f4-4b0c-4ea5-b783-bc82b58da0c1/Untitled.png)

Update value functions under the guidance of Bellman’s Equation (chart made by author)

Value iteration is actually guarantee to converge for finite state regardless of the initial states of the value (for proof please refer to the [Bible of RL](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)). If gamma the discount factor = 0, meaning we do not look beyond immediate rewards, then the value iteration will converge after 1 iteration. A smaller gamma leads to faster convergence as the consideration horizon is shorter, but typically not a better option for concrete problem solving. It is a factor to balance in engineering practice.

One might ask how does this work if all states are initialized as zero? Immediate reward in Bellman Equation is the key to bring in additional information and break the initial ice. We can think about the state that immediately lead to the goal state (assuming) Value propagate throughout state space like a virus. In plain language, make small wins, frequently.

![Value and policy functions interact until they converge to optimum together (source: [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/7d0f8276-f5d5-459a-8bd8-80a16bab2784/Untitled.png)

Value and policy functions interact until they converge to optimum together (source: [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf))

However Value iteration also suffers from its inefficiency. Value iteration requires to take the optimal action at each iteration by considering all actions. Value iteration demonstrates the feasibility as the vanilla approach (like Dijkstra) but typically not useful in practice. Policy iteration improves this by taking the action according to the current available policy, by Bellman Equation (Note NOT optimality equation). 

![The contrast of Bellman Equation and Bellman Optimality Equation (chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/f926a444-3563-475a-8883-57b3d69d1362/Untitled.png)

The contrast of Bellman Equation and Bellman Optimality Equation (chart made by author)

Policy iteration decouples policy evaluation and policy improvement. This is a much faster solution as each step is taken based on a given policy instead of exploring all possible actions and evaluate to find the action that maximizes the objective (despite that each iteration of policy iteration can be more computationally intensive than value iteration due to the policy evaluation step). In plain language, if you can only fully evaluate the consequence of one action, better use your own judgement and do your current own best.

## AlphaGo and MCTS — when nets meets trees

We have all heard the unbelievable story of AlphaGo beating the best human player in 2016. AlphaGo formulates the gameplay of Go as a MDP and solves it with Monte Carlo Tree Search (MCTS). Why not value iteration or policy iteration?

Value iteration and policy iterations are systematic, iterative method that solves MDP problems. Yet even with the improved policy iteration, it still have to perform time-consuming operation to update the value of EVERY state. A standard 19x19 Go board has roughly [2e170 possible states](https://senseis.xmp.net/?NumberOfPossibleGoGames). This vast amount of states will be intractable to solve with a vanilla value iteration or policy iteration technique.

AlphaGo and its successors use a [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) algorithm to find its moves guided by a value network and a policy network, trained on from human and computer play. Let's take a look at the vanilla MCTS first. 

![The four steps of MCTS by AlphaGo, combining both value network and policy network (source: [AlphaGo](https://www.nature.com/articles/nature16961), Nature 2016)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/119a5a5b-a2f1-48ab-b548-426b9d0a8b0b/Untitled.png)

The four steps of MCTS by AlphaGo, combining both value network and policy network (source: [AlphaGo](https://www.nature.com/articles/nature16961), Nature 2016)

**Monte Carlo Tree Search (MCTS)** is a method for policy estimation that focuses on decision-making from the current state. One iteration involves a four-step process: selection, expansion, evaluation, and backup.

During **selection**, the algorithm follows the most promising path based on previous simulations until it reaches a leaf node, which is a position not yet fully explored. In the **expansion** step, one or more child nodes are added to represent possible next moves. **Evaluation** or simulation involves playing out a random game from the new node until the end, known as a "rollout." Finally, during **backup**, the algorithm updates the values of the nodes on the path taken based on the game's result, increasing the value if the outcome is a win and decreasing it if it is a loss.

After a given number of iterations exhausted, we get percentage frequency with which immediate actions were selected from the root during simulations. During inference, the one with the most visit will be selected. Here is [an interactive illustration of MTCS](https://vgarciasc.github.io/mcts-viz/) with the game of tic-tac-toe for simplicity. 

MCTS uses two networks. The value network evaluates the winning rate from a given state. The policy network evaluates the action distribution for all possible moves from a given state. How do these two neural network make MCTS better? They are used to reduce the effective depth and breadth of the search tree: sampling actions using the policy network, and evaluating positions using the value network.

![The policy network and value network of AlphaGo (source: [AlphaGo](https://www.nature.com/articles/nature16961), Nature 2016)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/4ab14639-0e52-4caf-93dd-519278318372/Untitled.png)

The policy network and value network of AlphaGo (source: [AlphaGo](https://www.nature.com/articles/nature16961), Nature 2016)

Concretely, in **expansion** step, policy network samples most likely positions, pruning the breadth of search space. In **evaluation** step, value network gives instinctive scoring of the position, and a faster lightweight policy network does the rollout until game end to collect reward. MCTS used a weighted sum of the two to make finial evaluation.

> Note that a single evaluation of the value network also approached the accuracy of Monte Carlo rollouts using the RL policy network, but using 15,000 times less computation. This is very similar to a fast-slow system design, intuition vs reasoning, [system 1 vs system 2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) by Nobel laureate Daniel Kahneman (We can see similar design in more recent work such as [DriveVLM](https://arxiv.org/abs/2402.12289)).
> 

> To be exact, there seems to be two slow-fast systems in alphaGo, on different levels. On the macro level, policy network selects moves, faster rollout policy networks evaluates moves. On the micro level, the faster rollout policy network can be approximated by a value network which predict board position winning rate directly.
> 

What can we learn from AlphaGo for autonomous driving? AlphaGo demonstrates how to extract an excellent policy using a good world model (simulation). Autonomous driving similarly requires a highly accurate simulation to effectively leverage the AlphaGo algorithm. 

## MPDM (and successors) in autonomous driving

In the game of Go all states are immediately available to both game players (perfect information game) and essentially observation equals to state, and the game of Go is typically characterized by a MDP process. In contrast, autonomous driving is a POMDP process as the states can only be estimated through observation. 

POMDP connects perception and planning in a principled way. The normal solution of POMDP is very similar to MDP, with limited lookup ahead. The main challenging lies in the curse of dimensionality (or explosion in state space), and complex interaction with other agents. To make tractable progress in real time, domain specific assumptions are typically made to simplify the POMDP problem. [MPDM](https://ieeexplore.ieee.org/document/7139412) (and the [two](https://www.roboticsproceedings.org/rss11/p43.pdf) [follow-ups](https://link.springer.com/article/10.1007/s10514-017-9619-z), and [the white paper](https://maymobility.com/resources/autonomy-at-scale-white-paper/)) is one pioneering study in this direction.

[MPDM](notion://www.notion.so/mpdm.md) reduces POMDP to closed-loop forward simulation of a finite discrete set of semantic level policies, rather than performing evaluation for every possible control input for every vehicle (curse of dimensionality).

![Semantic actions helps control the curse of dimensionality (source: [EPSILON](https://arxiv.org/abs/2108.07993))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/a68d53ca-b602-4977-8aaf-b5467179fb1f/Untitled.png)

Semantic actions helps control the curse of dimensionality (source: [EPSILON](https://arxiv.org/abs/2108.07993))

The assumptions of MPDM. First, much of decision making made by human drivers is over discrete high level semantic actions (e.g. slowing, accelerating, lane-changing, stopping). This is referred to as a policy in this paper. The second implicit assumption is on other agents. Other vehicles will make reasonable safe decisions, and if the policy of a vehicle is decided, then the action (trajectory) is determined.

![The framework of MPDM (chart created by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/4c8d8cf0-b095-4579-97c9-583c7c978976/Untitled.png)

The framework of MPDM (chart created by author)

MPDM first select one policy for the ego vehicle out of many options (where the multipolicy in the name comes from), and select one policy for each nearby agents conditioned by the respective prediction, then performs forward simulation (like fast rollout in MCTS). The best interaction scenario after evaluation is then passed onto motion planning (such as [SCC](notion://www.notion.so/scc.md) mentioned in the joint spatiotemporal planning session).

MPDM can enable intelligent and human-like behavior of **active cut-in** into dense traffic flow even when there is not a sufficient gap present. This is NOT possible with a predict-then-plan pipeline which does not consider the interaction explicitly. Note that the prediction module is tightly integrated with behavior planning model through the forward simulation. 

MPDM assumes a single policy throughout the decision horizon (10s). MPDM essentially adopts a MCTS with one layer deep, and super wide considering all the possible agent predictions. This leaves huge room for improvement, and MPDM inspired many follow-up works such as EUDM, EPSILON and [MARC](https://arxiv.org/abs/2308.12021). For example,  EUDM considers more flexible ego policies and assigns a policy tree with depth of four, with a time duration of 2s for each policy over the decision horizon of 8s. To compensate for the extra compute induced by increased tree depth, EUDM does heavier width pruning by more efficient guided branching by identify critical scenario and key vehicles. This way, a more balanced policy tree is explored. 

Forward simulation of MPDM and EUDM used very simplistic driver model to perform forward simulation (IDM for longitudinal simulation, Pure pursuit for lateral simulation). MPDM also pointed out that the high fidelity realism matters less than the closed-loop nature itself. As long as the policy level decision is not affected by low level action execution inaccuracies.

![The conceptual diagram of decision making, where prediction, BP and MP integrates tightly (chart created by author)](https://cdn-images-1.medium.com/max/1600/1*bDd0arvG_Ndd11fwF9FkCg.png)

The conceptual diagram of decision making, where prediction, BP and MP integrates tightly (chart created by author)

Contingency planning in the context of autonomous driving refers to generating multiple potential trajectories that account for different possible future scenarios. A key motivating example is that experienced drivers anticipates multiple future scenarios and always plans for a safe backup plan. This leads to smoother driving experience, for example, even when cars performs sudden cut-ins into ego lane. A key aspect of this planning is deferring the decision bifurcation point, which means delaying the point at which different potential trajectories diverge. This delay allows the ego vehicle more time to gather information and respond to different outcomes, resulting in smoother and more confident driving behaviors like an experienced driver.

![Risk aware contingency planning (source: [MARC](https://arxiv.org/abs/2308.12021), RAL 2023)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/f99ea737-3a79-4e82-a7c7-c6af1f623f27/Untitled.png)

Risk aware contingency planning (source: [MARC](https://arxiv.org/abs/2308.12021), RAL 2023)

One possible drawback of MPDM and all other follow-up works still relies on simple policies designed for highway-like structured environments (such as lane keeping, lane change, etc). This may lead to limited capability to forward simulation to handle complex interaction. Following the example of MPDM, the key to make the POMDP is to simplify the action and state space through the growth of a high level policy-tree. It might be possible to create a more flexible policy tree, for example, by enumerate spatiotemporal relative position tags to all relative objects and then perform guided branching.

# Industry practices of decision making

Decision making is still a hot topic even in research recently. Even classical optimization methods have not been fully explored yet. ML methods could shine and have disruptive impact, especially with the advent of LLM, empowered by CoT or MCTS.

## Trees

Trees are more systematic ways to perform decision making. Tesla AI day 2021 and 2022 are heavily affected by AlphaGo. Tesla showcases the decision making capability to address highly complex interaction in the AI day.

High level, it follows the behavior planning (decision making) and then motion planning approach. As it searches for a convex corridor first then feed into continuous optimization. It is using spatiotemporal joint planning, from the narrow passing example showcased, a typical bottleneck for path-speed decoupled planning. 

![Neural network heuristics guided MCTS (source: [Tesla AI Day 2021](https://youtu.be/j0z4FweCy4M?t=4514))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/8373f95a-a070-4c2e-8c65-946c190c19b7/Untitled.jpeg)

Neural network heuristics guided MCTS (source: [Tesla AI Day 2021](https://youtu.be/j0z4FweCy4M?t=4514))

Hybrid system with data-driven and physics-based checks. Tesla also adopts MPDM-like approach. Starting with goals, generate seed trajectories, and evaluate key scenarios, then branch out to have more scenario variants, such as assert to or yield to a traffic agent. 

![An interaction search over policy tree (source: summarized from [Tesla AI Day 2022](https://youtu.be/ODSJsviD_SU?t=4074))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/b1ec4177-d72f-4d24-8bc4-b757abd2a50f/Untitled.png)

An interaction search over policy tree (source: summarized from [Tesla AI Day 2022](https://youtu.be/ODSJsviD_SU?t=4074))

One highlight of Tesla’s use of ML is the acceleration of tree search via trajectory optimization. For each node, use physics based optimization and neural planner. 10 ms vs 100 us. x10-x100 improvement. Neural network is trained with expert demos and offline optimizers.

Trajectory scoring is performed by combining classical physics-based checks (such as collision checks, comfort analysis) and neural network evaluators which predict intervention likelihood, and rate for human-likeness. The scoring helps prune the search space and helps the compute on the most promising outcomes.

Most people would argue that ML should be applied to high level decision making, but Tesla actually used ML in the most fundamental way to accelerate optimization and thus tree search.

The MCTS methods seems to be the ultimate tool to decision making. It seems that people studying LLM are trying to get MCTS into LLM, but people working on AD are trying to get rid of MCTS for LLM.

This is Tesla’s technology roughly two years ago. Since March of 2024, FSD switched to a more end-to-end approach which would be significantly different from here. .

## No trees

We can still consider interactions without implicitly trees. Ad-hoc logics can be implemented to perform one order of interaction between prediction and planning. Even one-order interaction between prediction and planning can already generate good behavior per TuSimple. MDPM in its original form is essentially one-order interaction, but done in a more principled and extendable way.

![Multi-order interaction between prediction and planning (source: [TuSImple AI day](https://www.bilibili.com/video/BV1Em4y1u7P7/?vd_source=73e03d3e246aa3ec284f028c7dcf0fa7), in Chinese, translated by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/90ba708f-38d8-4794-bf9d-7985c6a7e6e0/Untitled.png)

Multi-order interaction between prediction and planning (source: [TuSImple AI day](https://www.bilibili.com/video/BV1Em4y1u7P7/?vd_source=73e03d3e246aa3ec284f028c7dcf0fa7), in Chinese, translated by author)

TuSimple also demonstrated the capability to perform contingency planning, similar to that demonstrated in [MARC](https://arxiv.org/abs/2308.12021). 

![Contingency planning (source: [TuSImple AI day](https://www.bilibili.com/video/BV1Em4y1u7P7/?vd_source=73e03d3e246aa3ec284f028c7dcf0fa7), in Chinese, translated by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/b36086bd-a842-45fd-be37-027b7445155f/Untitled.png)

Contingency planning (source: [TuSImple AI day](https://www.bilibili.com/video/BV1Em4y1u7P7/?vd_source=73e03d3e246aa3ec284f028c7dcf0fa7), in Chinese, translated by author)

# **Self-Reflections**

After learning the basic building blocks of the classical planning system, including behavior planning, motion planning, and the principled way to handle interaction by decision making, I have had many self-reflection questions on what may be the potential bottlenecks of the system and how machine learning and neural networks (NN) may help. I am documenting my thinking process here for future reference and who may have similar questions. Note that information in this session may contain heavy personal bias and speculations.

## Why NN in planning?

Let’s look at the problem from three different perspective, in existing modular pipeline, as an end-to-end (e2e) NN planner or as e2e autonomous driving systems.

Going back to the drawing board, let’s review with the problem formulation of a planning system in an autonomous driving. The goal is to obtain a trajectory with safety, comfort and efficiency, under highly uncertain and interactive environment, with realtime engineering constraints onboard the vehicle. These factors are summarized as goals, environments and constraints in the chart below.

![The potentials of NN in planning (chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/5ab52b81-e9a7-499d-a1df-5856e564f710/Untitled.png)

The potentials of NN in planning (chart made by author)

Uncertainty can refer to uncertainty in perception (observation) and uncertainty in predicting long-term agent behaviors into the future. Planning also has to work with uncertainty in future trajectory predictions of other agents. As we discussed above, a proper way to do this is through a principled decision-making system. 

In addition planning has to accept more uncertain results from upstream from imperfect and sometimes incomplete perception, especially in the current age of [vision-centric and HD Map-less driving](https://towardsdatascience.com/bev-perception-in-mass-production-autonomous-driving-c6e3f1e46ae0?sk=8963783161435815fa1b0957fd325d39). Having [SD map](https://arxiv.org/abs/2403.10521) onboard as prior helps alleviate with this uncertainty, but it still can pose severe challenges to a heavy handcrafted planner system. This uncertainty with perception was considered a solved problem by L4 companies through heavy use of Lidar and HD Maps but has newly surfaced as the industry move to mass production autonomous driving solutions without these two crutches.

Interaction should be best treated with a principled decision making system such as a MCTS or a simplified version of MPDM. The main challenge is how to deal with the curse of dimensionality (or combinatorial explosion) by growing a balanced policy tree with smart pruning through domain knowledge of autonomous driving. MPDM and variants in academia and Tesla in industry introduced some good examples on how to grow this tree in a balanced way.

NN can also help with real time performance of planner by speeding up motion planning optimization. This can transfer the compute load from CPU to GPU and achieves orders of magnitude speed up. One order of magnitude faster optimization can have fundamental impact on the high-level algorithm design, such as MCTS.

Trajectories also needs to be more human like. Human likeness and takeover predictors can be trained with the (relatively) cheaply available huge amount of human driving data. It is also more scalable to increase the compute pool rather than maintaining an overgrowing army of engineering talents.

![The NN-based planning stack can leverage human driving data more effectively (Chart created by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/f796563d-ae61-4e67-8bd0-087ea49ba4a6/Untitled.png)

The NN-based planning stack can leverage human driving data more effectively (Chart created by author)

## What about e2e NN planners?

An e2e NN planner still constitutes a modular AD design, accepting structured perception results (and potentially latent features) as its input. This combines prediction, decision and planning into one single network. This is [what DeepRoute (2022) and Huawei (2024), among many others](https://www.xchuxing.com/article/124221), are claiming to do. Note that relevant raw sensor input such as navigation and ego vehicle information are omitted here. 

![A full autonomous driving stack with an e2e planner (chart made by author)](https://cdn-images-1.medium.com/max/1600/1*bkU7xzqKttz0HDaLK7owJg.png)

A full autonomous driving stack with an e2e planner (chart made by author)

This e2e planner can be also taken one step further to be an e2e autonomous driving system combining both perception and planning. This is what [Wayve LINGO-2 (2024)](https://wayve.ai/thinking/lingo-2-driving-with-language/) and Tesla FSDv12 (2024) are claiming to do. The benefits of this is first it solves perception issue. There are so many things that we cannot easily model explicitly yet with the commonly used perception interface. For example it is quite challenging to handcraft a driving system to [nudge a puddle of water](https://x.com/AIDRIVR/status/1760841783708418094), [slowing down for dips or potholes](https://x.com/AIDRIVR/status/1759843256513564997). Passing intermediate perception features will help but I doubt will not fundamentally resolve the issue.

In addition, emergent behavior will most likely help resolve corner cases in a more systematic fashion due to emergent behavior. The above smart behavior in handling the edge cases may have come from the emergent behavior of large models. 

![A full autonomous driving stack with a one-model e2e driver (chart made by author)](https://cdn-images-1.medium.com/max/1600/1*syl50VkFylmVUF4m9nonTw.png)

A full autonomous driving stack with a one-model e2e driver (chart made by author)

My heavy speculation is that in its ultimate form, the e2e driver would be a large vision and action-native multimodality model enhanced by a MCTS, if we are not bounded by compute. 

A world model in the context of autonomous driving, by most paper’s consensus as of 2024, is a multimodality model covering at least vision and action modes (or a VA model). Language is beneficial in that it can accelerates training, adds controllability and explainability, but it may not be a must. In its full blown form, a world model is a VLA (vision-language-action) model.

There are at least two ways to obtain a world model. One way is to train a video-native model by predicting future video frames, conditioned on or outputting accompanying actions, such as [GAIA-1](notion://www.notion.so/gaia1.md). The other way is to piggy back on a pretrained LLM as a starting point and add multimodality adaptors to it, such as [Lingo-2,](notion://www.notion.so/lingo2.md) [RT2](notion://www.notion.so/rt2.md) or [ApolloFM](https://mp.weixin.qq.com/s/8d1qXTm5v4H94HxAibp1dA). Such multimodality LLM are not native to vision or action. 

Such a world model can produce a policy itself via the action output and drive the vehicle directly. Alternatively, MCTS can query world model and use its policy acts to guide the search. This World Model-MCTS approach is much more computationally intensive and could have much higher ceiling than the direct World Model approach in terms of corner case handling due to the explicit reasoning logic. 

## Can we do without prediction?

Most current motion prediction modules represent future trajectories of agents other than ego as one or multiple discrete trajectories. It remains a question whether this prediction-planning interface is sufficient or necessary.

In a classical modular pipeline, prediction is still needed. Yet a predict-then-plan pipeline definitely caps the upper limit of autonomous driving systems, as we discussed above in the decision-making session. A more critical question is how to integrate this prediction module more effectively in the overall autonomous driving stack. Prediction should be used to help decision making, and a queryable prediction module by an overall decision-making framework is preferred, such as MPDM and variants. No severe issue with concrete trajectory as long as we integrated it correctly such as policy tree rollout.

Another issue with prediction is that open-loop KPIs, such as ADE (Average Displacement Error) and FDE (Final Displacement Error), are not effective metrics as they fail to reflect the impact on planning. Instead, metrics like recall and precision at the intent level should be considered.

In an end-to-end system, an explicit prediction module may not be necessary but implicit supervision among many other domain knowledge in a classical stack should definitely help or at least boost data efficiency of the leanring system. Evaluation of the prediciton behavior, explicit or implicit, will be helpful in debugging such an e2e system as well.

## Can we do with just nets but no trees?

Conclusions first. For an assistant, nets can achieve very high, even superhuman performance. For agents, I believe a tree is still beneficial (not necessarily a must).

First of all, trees can boost nets. Trees boost performance of a given network, NN based or not. In AlphaGo, although we have a policy network trained with supervised learning and reinforcement learning, the overall performance is still worse than the overall MCTS-based AlphaGo which integrates the policy network as one component. 

Second, nets can distill trees. Recall that in AlphaGo, MCTS used both value network and reward from a fast rollout policy network to evaluate a node (state, or board position) in the tree. It is also mentioned in the AlphaGo paper that we can also just do value function, but combining the results of the two gives best results. If we look closely the value network is essentially distilling the knowledge from the policy rollout by directly learning the state-value pair. It is very similar to how humans distills logical thinking of the slow system 2 into fast intuitive response of system 1. [Daniel Kahneman in his book](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) mentioned on example that a chess master who, after years of practice, can quickly recognize patterns and make rapid decisions that a novice would find challenging and require significant effort to achieve. This is exactly how the value network got trained in AlphaGo as a fast evaluation of a given board position.

![Grandmaster-Level Chess Without Search (source: [DeepMind, 2024](https://arxiv.org/abs/2402.04494))](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/63a6fd32-1734-4c72-9a04-428a6513a9b1/Untitled.png)

Grandmaster-Level Chess Without Search (source: [DeepMind, 2024](https://arxiv.org/abs/2402.04494))

Recent papers explore the upper limit of this fast system with nets. The [chess without search](https://arxiv.org/abs/2402.04494) paper shows that with enough data (prepared through tree search with a more conventional algorithm) we can achieve grandmaster level of proficiency, and there is a clear “scaling law” to this, with respect to data size and model size. 

So here we are with a power duo. Trees boost nets, and nets distills trees. This positive feedback loop is essentially what [AlphaZero](https://www.science.org/doi/full/10.1126/science.aar6404) uses to bootstrap itself to reach superhuman performance in multiples games.

Same principles apply for the development of LLM (large language models) as well. For games, since we have clearly defined rewards as wins or losses, we can use forward rollout to get the value of a certain action or state. For LLMs, there is no as clear from the game of Go, so we have to rely on human preferences to rate the models via RLHF (reinforcement learning with human feedbacks). Yet with ChatGPT trained in place, we can use SFT (supervised finetuning, essentially imitation learning) to distill smaller yet also powerful models without RLHF. 

Going back to the original question, nets can achieve extremely high performance with large quantities of high quality data. This could already be a good enough assistant, depending on the tolerance of errors, but may not be good enough for an agent. For a system targeting as driving assistance (ADAS), nets via imitation learning may be good enough. 

Trees can greatly boost the performance of nets with an explicit reasoning loop, which is perhaps more suitable for an fully autonomous agent. Of course how heavy the tree or reasoning loop depends on the return on investment of the engineering resource. For example, one order of interaction will already have great benefit as shown in [TuSimple AI Day](https://www.bilibili.com/video/BV1Em4y1u7P7/?vd_source=73e03d3e246aa3ec284f028c7dcf0fa7). 

## Can we use LLMs to make decisions?

From the summary below of the hottest representatives of AI systems, we can see that LLMs are not designed to perform decision making. In essence, LLMs are trained to complete document, and even SFT-aligned LLM assistants essentially treat dialogues as a special type of document (completing a dialogue record). 

|  | Representatives | Interaction | Clearly defined rewards |
| --- | --- | --- | --- |
| The game of Go | AlphaGo | Yes | Yes |
| LLM | ChatGPT | No | No |
| Autonomous Driving | Tesla FSD v12 | Yes | No |
| Robotics | TeslaBot | No (very weak) | No |

I do not quite agree with recent claims that LLMs are slow systems (system 2). They are unnecessarily slow in inference due to hardware constraints, but in the vanilla form LLMs are fast systems as they could not do counterfactual check. Prompting techniques such as CoT (chain of thought) or ToT (tree of thoughts) are actually a simplified form of MCTS, and they make LLMs a slower system. 

Tons of research along this line trying to marry a full-blown MCTS with LLM. Specifically, [LLM-MCTS](https://arxiv.org/abs/2305.14078) (NeurIPS 2023) treats LLM as a commonsense “world model” and use LLM-induced policy acts as a heuristic to guide the search. LLM-MCTS outperforms both MCTS alone and policies induced by LLMs by a wide margin, for complex, novel tasks. The [highly speculated Q-star](https://www.lesswrong.com/posts/JnM3EHegiBePeKkLc/possible-openai-s-q-breakthrough-and-deepmind-s-alphago-type) from OpenAI seems to be along the same line of boosting LLM with MCTS, as the name suggests.

# The trend of evolution

Below is a rough evolution of planning stack in autonomous driving. It is rough as the ones below are not necessarily more advanced than the one above, and their debut may not be in the exact chronological order. Still we can observe the general trends. Note that the listed representative solutions from the industry are based on my interpretation of various press releases and I could be way off. 

One trend is that we to see a more end-to-end design with more modules consolidated into one. We can see the stack evolves from path-speed decoupled planning to joint spatiotemporal planning, and from predict-then-plan system to a joint prediction and planning system. Another trend is that we see more and more machine learning based components come into play, especially in the last three rows. These two trends converge towards an end-to-end NN-planner (without perception) or even an end-to-end NN driver (with perception)

![A rough history of evolution of planning (Chart made by author)](https://prod-files-secure.s3.us-west-2.amazonaws.com/8934b82c-ccb5-4320-88cd-dd48fd656172/87a41ead-d2e6-49e3-9a3d-45acec616d6e/Untitled.png)

A rough history of evolution of planning (Chart made by author)

# Takeaways

- Know your problems and know your tools. In autonomous driving, the problem setting is that we have decision making and motion planning modeled as POMDP in its full blown form. For tools, we have the trioka of planning (search, sampling, optimization), and solutions to decision making (value iteration, policy iteration, MCTS, MPDM, and of course, machine learning).
- ML is a tool, not a solution. ML can help with planning even in current modular design.
- Start with a full formulation, then make reasonable assumptions to simply the problem to balance between performance and resource. This will help creating a clear north star, which will guide toward a more future-proof system design and allow improvements once more resource becomes available. Recall from POMDP’s formulation to engineering solutions like AlphaGo’s MCTS and MPDM.
- Theoretically beautiful algorithms are great for understanding the concept (such as Dijkstra and Value iteration) but they have to be heavily adapted for engineering practices (Value iteration is to MCTS as Dijkstra's algorithm is to hybrid A-star).
- Planning has strong toolkit in resolving deterministic (not necessarily static) scenes. Decision making in stochastic scene is the most challenging task toward full autonomy.
- Contingency planning can help merge multiple futures into a common action. It is good to be aggressive to the degree that you can always resort to your backup plan.
- Whether e2e one model can solve full autonomy remains unclear. It may also need the help from classical methods such as MCTS. Nets can do assistants, trees can do agents.

# Acknowledgements

- This blog post is heavily inspired by [Wenchao Ding](https://wenchaoding.github.io/personal/index.html)’s course on planning on [Shenlan Xueyuan (深蓝学院)](https://www.shenlanxueyuan.com/course/671).
- Heavy discussion with [Naiyan Wang](https://winsty.net/) and Jingwei Zhao. They also gave critical feedbacks to the initial draft. Thanks to critical feedbacks from [论文推土机](https://www.zhihu.com/people/george-reagan). Thanks for insightful discussion with [Professor Wei Zhan](https://zhanwei.site/) from Berkeley in trends in academia.

# Reference

- [End-To-End Planning of Autonomous Driving in Industry and Academia: 2022–2023](https://arxiv.org/abs/2401.08658), Arxiv 2024
- [BEVGPT: Generative Pre-trained Large Model for Autonomous Driving Prediction, Decision-Making, and Planning](https://arxiv.org/abs/2310.10357), AAAI 2024
- [Towards A General-Purpose Motion Planning for Autonomous Vehicles Using Fluid Dynamics](https://arxiv.org/abs/2406.05708)
- [Tusimple AI day](https://www.bilibili.com/video/BV1Em4y1u7P7/?vd_source=73e03d3e246aa3ec284f028c7dcf0fa7), in Chinese with English subtitle on Bilibili, 2023/07
- [Tech blog on joint spatiotemporal planning by Qcraft](https://zhuanlan.zhihu.com/p/551381336), in Chinese on Zhihu, 2022/08
- [A review of entire autonomous driving stack](https://zhuanlan.zhihu.com/p/53495492), in Chinese on Zhihu, 2018/12
- [Technical blog on ApolloFM](https://mp.weixin.qq.com/s/8d1qXTm5v4H94HxAibp1dA), in Chinese by Tsinghua AIR, 2024
- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.semanticscholar.org/paper/Optimal-trajectory-generation-for-dynamic-street-in-Werling-Ziegler/6bda8fc13bda8cffb3bb426a73ce5c12cc0a1760), ICRA 2010
- [MP3: A Unified Model to Map, Perceive, Predict and Plan](https://arxiv.org/abs/2101.06806), CVPR 2021
- [NMP: End-to-end Interpretable Neural Motion Planner](http://www.cs.toronto.edu/~wenjie/papers/cvpr19/nmp.pdf), CVPR 2019 oral
- [Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711), ECCV 2020
- [CoverNet: Multimodal Behavior Prediction using Trajectory Sets](https://arxiv.org/abs/1911.10298), CVPR 2020
- [Baidu Apollo EM Motion Planner](https://arxiv.org/abs/1807.08048), Baidu, 2018
- [AlphaGo: Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961), Nature 2016
- [AlphaZero: A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/full/10.1126/science.aar6404), Science 2017
- [MuZero: Mastering Atari, Go, chess and shogi by planning with a learned model](https://www.nature.com/articles/s41586-020-03051-4), Nature 2020
- [ToT: Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601), NeurIPS 2023 Oral
- [CoT: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), NeurIPS 2022
- [LLM-MCTS: Large Language Models as Commonsense Knowledge for Large-Scale Task Planning](https://arxiv.org/abs/2305.14078), NeurIPS 2023
- [MPDM: Multipolicy decision-making in dynamic, uncertain environments for autonomous driving](https://ieeexplore.ieee.org/document/7139412), ICRA 2015
- [MPDM2: Multipolicy Decision-Making for Autonomous Driving via Changepoint-based Behavior Prediction](https://www.roboticsproceedings.org/rss11/p43.pdf), RSS 2015
- [MPDM3: Multipolicy decision-making for autonomous driving via changepoint-based behavior prediction: Theory and experiment](https://link.springer.com/article/10.1007/s10514-017-9619-z), RSS 2017
- [EUDM: Efficient Uncertainty-aware Decision-making for Automated Driving Using Guided Branching](https://arxiv.org/abs/2003.02746), ICRA 2020
- [MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving](https://arxiv.org/abs/2308.12021), RAL 2023
- [EPSILON: An Efficient Planning System for Automated Vehicles in Highly Interactive Environments](https://arxiv.org/abs/2108.07993), TRO 2021

# Appendix

Motion Prediction

Now coming back to motion prediction, we should have a better understanding and to its requirements. 

Prediction is essentially doing plans for other cars as well, but we may not need to do as heavy or as good a job as the planner for ego (but good enough to coach or guide ego planning). A super lightweight planner, 16, 32 or 64 agents. A typical ego motion planner based on above techniques are too heavy.

## Ego-centric prediction

No interaction at all. Model-based or learning based. 

A baseline motion prediction model are purely based on kinematics assume a constant velocity (CV) or constant turn rate (CT). They work 2-3 seconds at most. For longer term (8s+),  intension is the key! Even in learning based methods, intention is still a very good abstraction to leverage in network design. 

Intention can be predicted by creating manual rules, learning by simple SVM models, or by DL models. Most of the DL-based models are ego-centric before 2021. Typical works include TNT, denseTNT, and HiVT.

## Scene-centric prediction

Since interaction between multi-agent is critical since 2021 more and more scene-centric prediction networks starts to appear. Such as SceneTransformer, WayFormer and QCNet. 

Still no interaction between planning and prediction by default, but they can be modified to perform prediction and planning joint rollout.