# Continous-Control with Deep Deterministic Policy Gradient
This implementation is based on the Deepmind paper [Continuous control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf). In this report, we will go through the implementation details and hyperparameters selection.

## Environment
![alt text](https://github.com/vjrahil/Continous-Control/blob/master/Images/Environment.png) <br />
### Details
The following environment is provided by the [Unity ML-toolkit](https://github.com/Unity-Technologies/ml-agents). Here are some of the details of the envrionment. <br />
* *Set-up*: Double-jointed arm which can move to target locations.
* *Goal*: The agents must move its hand to the goal location, and keep it there.
* *Agents*: The environment contains 10 agent with same Behavior Parameters.
* *Agent Reward Function (independent)*: <br />
  * +0.1 Each step agent's hand is in goal location.
* *Behavior Parameters*:
  * *Vector Observation space*: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  * *Vector Action space*: (Continuous) Size of 4, corresponding to torque applicable to two joints.
  * *Visual Observations*: None.
* *Benchmark Mean Reward*: 30

### Real World Scenario
Some of the researchers were able to train a similar task on real world robot. This type of task can be used to track objects in real life. Here is the [video](https://www.youtube.com/watch?v=ZVIxt2rt1_4) and [paper](https://arxiv.org/pdf/1803.07067.pdf) regarding it.<br />

### Downloading the environment
* *Linux*: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* *Mac OSX*: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* *Windows (32-bit)*: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* *Windows (64-bit)*: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Architecture
The DDPG algorithm uses both the policy network(Actor) and a value based network(Critic). It combines both these network to create a more stable algorithm for tackling contiuous action space problems.
The idea behind using the Critic network is to help the policy network(actor) to reduce its variance. The critic network has low variance and low bias, on the other hand, the actor-network has high variance, so with the help of critic, we try to reduce it.
### Actor Network
* The actor-network takes in states and generates a vector of four, each in the range of (-1,1). For this, we use tanh to keep the final output in this range.
* We use two copies of this network,a local and a target network. This helps in stablising the loss function.

|Layers|Dimensions|Activation Function|
|--------|---------|------------------|
|Linear|(state_size,256)|Relu|
|Linear|(256,128)|Relu|
|Linear|(128,action_size)|Tanh|

### Critic Network
* The critic network takes in states and the actions for each state generated from the Actor-network and uses them to produce a value.
* We use two copies of this network, a local and a target network. This helps in stabilising the loss function.

|Layers|Dimensions|Activation Function|
|------|----------|--------------------|
|Linear|(state_size,256)|Leaky Relu|
|Linear|(256 + action_size,128)|Leaky Relu|
|Linear|(128,1)|None|

### Exploration
In discrete action space, we use a probabilistic action selection policy(epsilon-greedy). But it won’t work in a continuous action space. In the paper of [Continuous control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf), the aurthors used [Ornstein Uhlenbeck Process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) to add noise to the action. This process uses the previous noise to prevent the noise from canceling out or “freezing” the overall dynamics.

### Memory
To store the memory, I used a Replay Buffer of size *int(1e6)*. All the experiences were stored as a tuple of the form *(state,action,reward,next_state,done)*
## Hyperparameter

|Parameters|Values|
---------|--------|
|Actor LR| 1e-4|
|Critic LR| 1e-4|
|TAU| 1e-3|
|BATCH_SIZE| 128|
|BUFFER SIZE| int(1e6)|
|GAMMA| .99|
|Actor optimizer| Adam|
|Critic optimizer|Adam|
|Weight DECAY for Critic optimizer| 0.0|  
|T_STEPS|20|
|N_LEARN_UPDATES|10|

T_STEPS specifies the number of timesteps after which the networks were updated.<br />
N_LEARN_UPDATES specifies the number of times the networks were updates at each updating timestep.<br />
## Results

After a lot hyperparameter tuning I was able to solve the environment in 38 episodes. Here is the episode vs score graph.<br >
![alt text](https://github.com/vjrahil/Continous-Control/blob/master/Images/Result.png)

## Experimentations
* Try out different algoritm such as [Trust Region Policy Optimization(TRPO)](https://arxiv.org/pdf/1502.05477),[Proximal Policy Optimization(PPO)](https://arxiv.org/abs/1707.06347), [Distributed Distributional Deterministic Policy Gradients(D4PG)](https://openreview.net/forum?id=SyZipzbCb) 
* Implement a Priority Experience Replay Buffer.
* As fine tuning the hyperparameter took a very long time, we can use algorithm such as grid search etc, to do that for us.

## Repository Instructions
* Clone the directory by using this command.
```
    git clone https://github.com/vjrahil/Continous-Control
```
