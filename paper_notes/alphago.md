# [AlphaGo: Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)

_June 2024_

tl;dr: MCTS trees with neural nets bets human in the game of Go.

#### Overall impression
Value iteration and policy iterations are systematic, iterative method that solves MDP problems. Yet even with the improved policy iteration, it still have to perform time-consuming operation to update the value of EVERY state. A standard 19x19 Go board has roughly [2e170 possible states](https://senseis.xmp.net/?NumberOfPossibleGoGames). This vast amount of states will be intractable to solve with a vanilla value iteration or policy iteration technique.

AlphaGo and its successors use a [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) algorithm to find its moves guided by a value network and a policy network, trained on from human and computer play. Both the value network and policy network takes in the current state of the board and produces a singular state value function V(s) of the board position and the state-action value function Q(s, a) of all possible moves given the current board position. Neural networks are used to reduce the effective depth and breadth of the search tree: evaluating positions using a value network, and sampling actions using a policy network.

Every leaf node (an unexplored board position) in the MCTS is evaluated in two very different ways: by the value network; and second, by the outcome of a random rollout played out using the fast rollout policy. Note that a single evaluation of the value network also approached the accuracy of Monte Carlo rollouts using the RL policy network, but using 15,000 times less computation. This is very similar to a fast-slow system design, intuition vs reasoning, [system 1 vs system 2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) by Nobel laureate Daniel Kahneman (We can see similar design in more recent work such as [DriveVLM](https://arxiv.org/abs/2402.12289)).

#### Key ideas
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



#### Technical details
- Summary of technical details, such as important training details, or bugs of previous benchmarks.

#### Notes
- Value iteration is to MCTS as Dijkstra's algorithm is to (hybrid) A-star: both value iteration and Dijkstra's systematically consider all possibilities (covering the entire search space) without heuristics, while MCTS and A* use heuristics to focus on the most promising options, making them more efficient for large and complex problems. 
- Question: PN is essentially already e2e, why need VN and MCTS?
    - My guess: Small scale SL generate PN not strong enough, so need RL and MCTS to boost performance.
    - E2E demonstrates that with enough data, e2e SL can generate strong enough PN itself.
    - Maybe later MCTS will come back again to generate superhuman behavior for driving.

