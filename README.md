following the og book - suttonBartoIPRLBook2ndEd.pdf
i really like this sheet for all the policy gradient algos - https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#ppo
this for ppo - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details

---

following are my obsidian notes while studying RL from Sutton and Barto.

## Chapter 1

| Reinforcement Learning                                                    | Supervised Learning                                       | Unsupervised Learning                                     |
| ------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| Gets to decide implicitly the next state of the environment.              | Doesn't get to decide at all which data points to sample. | Doesn't get to decide at all which data points to sample. |
| Learns to predict what the next state would be, in conjunction to action. | Only learns the label and, as a consequence, the loss.    | Predicts structure in un-labelled data.                   |
| Does not rely on examples of correct behavior.                            | Relies on labelled examples.                              | Does not rely on examples of correct behavior.            |

---
## Chapter 3: Finite MDPs

Policy: $\pi(a | s)=Pr\{A_t=a|S_t=s\}$
Return: $G_t=\sum_{k=0}^{T-t-1}\gamma^kR_(t+k+1)$

An RL task that satisfies the Markov Property is called Markov Decision Process.
Transition Probability Function: $p(s', r | s, a)$ = $Pr\{S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a\}$
#### Value Functions
Value of a *state* $s$ under policy $\pi$ : $v_\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]$ - State value function
Value of an *action* $a$ under policy $\pi$ : $q_\pi(s,a)=\mathbb{E}_\pi[G_t|S_t=s, A_t=a]$  - Action value function

#### Bellman Equation for $v_\pi$
Expresses relation between the value of a state and the value of its successor states. Just draw backup diagram of $v_\pi$.

$v_\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]$

#### Better Policy
$\pi\geq\pi'$ iff $v_\pi(s)\geq v_{\pi'}(s),\forall s\in S$

#### Optimal Value Functions
$v_*(s)=max_\pi v_\pi(s)$ : The maximum state value of state s under any policy $\pi$
$q_*(s,a)=max_\pi q_\pi(s,a)$

#### Bellman Optimality Equation for $v_\pi$

$v_*(s)=\max_{a\in A(s)}\sum_{s',r}p(s',r|s,a)[r+\gamma v_*(s')]$

---
## Chapter 4: Dynamic Programming

These methods use other state-values/action-values to update the estimates.
### 4.1 Policy Evaluation (E)
This calculates the state-value $v_\pi(s)\forall s\in S$ for any arbitrary policy $\pi$

**Iterative Policy Evaluation**: Algorithm to compute state-values for all states, until convergence of an arbitrary policy.

$v_k(S)$ is a vector of all state values at $k-th$  iteration.
$v_{k+1}(S)=T^\pi v_k(S)$ , where $T^\pi$ is the Bellman Expectation Operator. (Uses the Bellman Equation for state-value)

### 4.2 Policy Improvement (I)
Q. How to improve a policy, once it is evaluated. 
A. At state s, choose another $a'$ as a part of new policy $\pi$ s.t. $a'\neq \pi(s)$. `Greedy.`
Q. When to choose?
A. Whenever $q(a',s)\geq v_\pi(s)$. Then the new policy would be overall Better than the old policy. This result is called the **Policy Improvement Theorem**. 

### 4.3 Policy Iteration (PI)
$\pi_0\overset{E}{\rightarrow}v_{\pi_0}\overset{I}{\rightarrow}\pi_1\overset{E}{\rightarrow}v_{\pi_1}\overset{I}{\rightarrow}\pi_2$ 

### 4.4 Value Iteration (VI)
In 4.1 we used Bellman equation for state-value to update the state value. But what if we used the Bellman optimality equation to update it instead?
We would converge to $v_*$. Value Iteration iterates over the value function directly. Faster than PI.

Q. Must we wait for exact convergence of step $E$ in 4.3?
A. Shit, no.

### 4.6 Generalized Policy Iteration (GPI)
Do 4.3, but do E for a single state. This is asynchronous DP. Shit doesn't need a full sweep in the E step.

==Bottleneck Alert:== These are DP methods and all require $p(s',r|s,a)$.

---
## Chapter 5: Monte Carlo Methods
These methods wait till the end of the episode and update the state-value/action-value with the actual return.

If we don't have access to transition probabilities, we cannot simulate and weight using $p$. We have to play out. How do we do that?

### 5.1 MC Prediction
Need $\pi$ - a policy to be evaluated, V - an arbitrary state-value vector. 
Generate an episode using $\pi$. Log return $G$ for the first-visit to a state s. Update V. Do for many episodes.

Therefore we can keep on refining the state-value $v_\pi$

To update policy, we must do:
$\pi'(s)=\underset{a}{argmax}[q_\pi(s,a)]=\underset{a}{argmax}\sum_{s',r}p(s',r|a,a)[r+\gamma v_{\pi}(s')]$
==Now, if we do not have a model for the environment, policy improvement step necessitates that we choose to track $q$ instead of $v$.==

To estimate action-values: is called *MC evaluation.*
### 5.3 MC Control 
Alternate MC evaluation with a greedy policy-improvement step + guarantee some initial exploration by starting episodes in random state-action pairs.

MC ES: Do {MC Evaluation + PI episode by episode} + Exploring Starts

### 5.4 MC Control w/o Exploring Starts
1. On-policy Methods:
   Enforce $\epsilon-greedy$ policies i.e. $\pi(a|s)\geq \frac{\epsilon}{|A(s)|}$ instead of greedy policy. 
2. Off-Policy Methods:
   $\pi$: Target Policy, $\mu$: Behavior Policy.
   Assumption of Coverage: $\pi(a|s)\geq0\implies\mu(a|s)$

### 5.5 Off-policy Prediction via Importance Sampling

$\underset{x\sim p(x)}{\mathbb{E}}[f(x)]$ is estimated using MC sampling. We need N samples of x, drawn from p(x).
Or, we could sample from q(x), and estimate $f(x)\frac{p(x)}{q(x)}$.

Here, x is an episode. 
p(x) is Pr{episode generated under $\pi$}. 
q(x) is Pr{episode generated under $\mu$}.
f(x) is the Return from episode x.

==Trick:== $\frac{p(x)}{q(x)}$ can be expressed in terms of $\pi$ and $\mu$ alone, not needing transition probability because it cancels out in fraction.

---
## Chapter 6: Temporal-Difference (TD) Learning
is an algorithm for Policy Evaluation.

|                              | MC Method [constant-$\alpha$ MC]                                      | TD Method [TD(0)]                                                                                         |
| ---------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Update Rule**              | $V(S_t)\leftarrow V(S_t) + \alpha[G_t-V(S_t)]$                        | $V(S_t)\leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]$                                      |
| **Update Freq.**             | Must wait till end of episode.                                        | Updates on the next step.                                                                                 |
| **Bootstrapping**            | No                                                                    | Yes                                                                                                       |
| **Nature of Update**         | Each estimate state/action-value shifts towards the estimated return. | Each estimate state/action-value shifts towards the estimate that immediately follows it.                 |
|                              | Off-line                                                              | On-line: advantageous when applications have really long episodes.                                        |
| Environment Transition Prob? | Model-free                                                            | Model-free                                                                                                |
| In batch-training            | Minimizes the MSE on the training data.                               | Maximizes the likelihood of the training data.                                                            |
| Bias-Variance Tradeoff       | High Variance, and unbiased.                                          | Introduces small Bias, but has smaller variance. Bias$^2$ + Variance is lower hence MSE is overall lower. |
TD(0) converges to the $certainty-equivalence\space estimate$.
- [ ] MC's estimate $\hat{V}_*(s)=\sum_{i=1}^k G(s)$
- [ ] TD(0)'s estimate $\hat{V}_*(s)=\hat{R}+\gamma\hat{P}(s'|s)\hat{V}_*(s')$. Where $\hat{P}(s'|s)=\frac{N(s\rightarrow s')}{N(s)}$ and $\hat{R}(s)$ is just the average of all rewards after state s in the batch.

TD(0) is implicitly using the using the assumptions of the data generating process (transition probability $p$) being a Markov Reward Process (MRP). 

MRP = MDP + Fixed Policy. MRPs are useful for long-term evaluation of the policy.

Constant-$\alpha$ MC is identical to TD($\infty$).

### 6.4 SARSA
is an on-policy TD control PGI algorithm. That iteratively updates Q values and policy $\pi$.
$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1, A_{t+1}}) - Q(S_t,A_t)]$

### 6.5 Q-Learning
is an off-policy TD control GPI algorithm.
$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma \underset{a}{max} Q(S_{t+1}, a) - Q(S_t,A_t)]$

The effect is of directly estimating $q_*$ irrespective of the policy $\pi$. However, $S_{t+1}$ is derived using the policy $\pi$.

---

## Chapter 8: Planning & Learning w/ Tabular Methods
- Planning Methods: Require Model of the Environment: Heuristic Search, DP.
- Learning Methods: Model Free: MC and TD.

- Environment Model: Distribution | Sample.
- Simulated Experience: When models are used to generate a single episode (using sample models) or generate all possible episodes (using distribution models).  

$$
model \quad \overset{planning}{\rightarrow} \quad \pi
$$
- Planning $\rightarrow$ [1] State-space Planning, [2] Plan-space Planning.
- State-space Planning:
$$ 
model \rightarrow simulated \space experience \overset{backups}{\rightarrow} values \rightarrow \pi
$$
Heart of both *learning and planning* methods is estimation of value functions using backup methods.

We can actually take 6.5 Q-learning and have it generate episodes from a sample model of the environment. In which case its called Q-planning. 

### 8.2 Integrating Planning, Acting & Learning

Planning can be done on-line. i.e. model can be learnt while interacting with the environment and it can be used to improve the policy. Hence there are two roles for real experience:
[1] model-learning: improve the model
[2] direct RL: directly improve the value-function (using MC, TD(0), SARSA, Q-learning)

![[Pasted image 20250514101838.png]]
Basically this sums up the Dyna-Q algorithm. 

### 8.4 Prioritized Sweeping
Vanilla Dyna-Q algo does Q-planning by sampling from the Model(S,A) at random. 
Prioritized sweeping algorithm adds those (S,A) pairs to a priority queue whose update value $[R+\gamma \underset{a}{max}Q(S',a)-Q(S,A)]$ is above a certain $\theta$. We can then sample the important (S,A) pairs from the past experience.

### 8.7 Heuristic Search
All state-space planning methods in AI are collectively called *heuristic search.*
Actually we go over a tutorial: [Searching for Solutions.](https://artint.info/3e/html/ArtInt3e.Ch3.html)

$\rightarrow MCTS$ The book doesn't mention this. Moving Ahead
## Chapter 9: Approximate Solution Methods

==Aims to give a method on how to solve for continuous state space or a huge state space==
Using function approximation: $\hat{v_\pi}(s, \textbf{w}) \approx v_\pi(s)$.

So we parametrize the value functions $v\space and \space q$.

### 9.2 On-line gradient-descent TD($\lambda$) for estimating $v_\pi$
#### Algorithm
a. initialize **w**
b. repeat (for each episode):
c.    **e**=0
d.    S $\leftarrow$ initial state of the episode
e.    repeat (for each step of episode):
f.         A $\leftarrow$ $\pi(S)$ # we are actually evaluating this policy and hence assume that it somehow gives us actions. This is not a control setup.
g.        execute A, observe reward R, and next state S'
h.        $\delta \leftarrow R + \gamma \hat{v}(S', w) - \hat{v}(S, w)$
i.         e $\leftarrow \gamma\lambda e + \nabla\hat{v}(S, w)$ 
j.         w $\leftarrow w+ \alpha \delta e$
k.        S$\leftarrow$S'
l.    until S' is terminal

This is an algo for policy evaluation. This can be extended to control as well. These methods are all called **action-value methods.**


## Chapter 13: Policy Approximation

Here we parametrize the policy. And we shall update the policy parameters using a performance  $J(\theta)$.

$$
\theta_{t+1} := \theta_{t} + \alpha \nabla_{\theta}\hat{J(\theta_{t})}
$$
where $\hat{J(\theta_t)}$ is some stochastic estimate. All methods following this general schema are policy-gradient methods. If we learn a value function in addition to a parametrized policy, we call them actor-critic algorithms. 

### 13.2 The Policy Gradient Theorem

In an episodic case the value of performance measure of a policy $J(\theta)$ is just $v_{\pi_\theta}(s_0)$, i.e. the expected return at state $s_0$.

$$
\nabla_\theta J(\theta) = \nabla_\theta v_{\pi_\theta} (s_0) \propto \underset{s \in S}{\sum} \mu(s) \underset{a \in A}{\sum} q_\pi(s,a) \nabla_\theta \pi(a|s; \theta)
$$
Now if we sample an episode on-policy, we are only concerned with $q_\pi(s,a)$ and $\pi$ itself. Which we have access to. 

The expectation on the RHS becomes:
$$
\underset{\pi}{\mathbb{E}} \left[\underset{a \in A}{\sum} q_\pi(s,a) \nabla_\theta \pi(a|s; \theta)\right]
$$

We have removed the second summation owing to the fact that $\underset{a}{\sum}\pi(a|s)=1$

$$
= \underset{\pi}{\mathbb{E}} \left[q_\pi(s,a) \nabla_\theta \{ln\space\pi(a|s; \theta)\}\right]
$$
$$
= \underset{\pi}{\mathbb{E}} \left[G_t .\nabla_\theta \{ln\space\pi(a|s; \theta)\}\right]
$$
#### REINFORCE Algo:
$$
\theta_{t+1} := \theta_{t} + \alpha \left[G_t .\nabla_\theta \{ln\space\pi(a|s; \theta)\}\right]
$$

This has high variance and slow convergence. Fix? Baseline.
See that:
$$
\underset{a}{\sum} b(s).\nabla_\theta\pi(a|s;\theta) = 0
$$
so long as $b$ is not a function of a. 
#### REINFORCE with baseline:
$$
\theta_{t+1} := \theta_{t} + \alpha \left[(G_t-b(S_t)) .\nabla_\theta \{ln\space\pi(a|s; \theta)\}\right]
$$
$b(S_t)$ can be $v(S_t, w)$,
REINFORCE is a MC method for updating policy parameters. We can use MC estimates to update the state function. 
Key properties of using state values as baseline:
1. $\mathbb{E}[G_t]=\mathbb{E}[v(S_t, w)]$ i.e. unbiased.
2. $var(G_t-v(S_t, w))\leq var(G_t)$ i.e. variance reduction. We are pushing $\theta$ in proportion to what is expected at that state and not just due to a large $G_t$. 

but how do we get the value of $v(S_t, w)$? train a net :)
### 13.5 Actor-Critic Methods
knowing the value function can assist the policy update, such as by reducing gradient variance in vanilla policy gradients.

If we bootstrap the state-value function in REINFORCE w/ baseline, we have an actor-critic method.
The TD-error $\delta$ is used to update both $w$ and $\theta$.
Instant on-line critique to the policy. 

So intuitively, $\delta = R_t + \gamma v(S_{t+1,w}) - v(S_t, w)$ measure how off my current value estimate is w.r.t. the future discounted value estimate. [Both are expected returns, but latter is said to be a tid-bit better.]
$w_{t+1}:=w{t} + \alpha \delta \nabla v(S_t,w)$ : Nudge my state function parameters $w$ in proportion to this TD error.
$\theta_{t+1} := \theta_t + \alpha \delta \nabla(ln(\pi (a|S_t;\theta)))$: Nudge my policy parameters $\theta$ with the information that the action that was just performed that produced such TD error was good or bad.

==The gradients give me information on how sensitive my action / state-value is to each policy / state function parameter. ==

---
Policy Gradient Algos: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

Advantage: How good is taking an action $a$ at a state $s$?
$A(s, a) = Q(s, a) - V(s)$.
Few ways to estimate advantage:
1. $\hat{A}(s_t,a_t) = G_t-V(s_t)$ $-$ *Monte-Carlo estimate*
2. $\hat{A}(s_t, a_t)=\delta_t=r_t+\gamma V(s_{t+1}) - V(s_t)$ $-$ *TD error*
3. $\hat{A_t}^{GAE(\lambda, \gamma)}=\sum^{\infty}_{l=0}(\gamma \lambda)^l \delta_{t+l}$ $-$ *Exponential average of past $A$ estimates: Generalized Advantage Estimation (GAE)*

---
#### Proximal Policy Optimization
I'd implemented actor-critic method using Q-value net and a policy net.
To convert this to PPO, following changes were made:
1. Q-value net $\rightarrow$ V-value net
2. Instead of updating the $\theta$ and $w$ after each step, we save all states, rewards, actions, v-values, action log-probs in a rollout buffer.
3. Need to compute advantages and returns.

Note: in TRPO the importance sampling ratio in the $J(\theta)$ expression is not the trust region. the trust region is the constraint of K.L. between $\pi_\theta$ and $\pi_{\theta_{old}}$. in ppo we have no constraint but a surrogate clipped objective. the clipping is to ensure trust region.
