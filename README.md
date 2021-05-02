# Udacity Deep Reinforcement Learning Project: Tennis
## Project Overview 
 In this project, I experimented with the Unity ML-Agents Tennis environment. 
 I used MADDPG algorithm in my project [(paper link)](https://arxiv.org/abs/1511.05952). 
 This is a variant of the DDPG algorithm for multi-agent environments.

#### Project Files
- **sovler.py:**  This file contains the maddpg() method which is used to train the RL agent  
- **agent.py:**  This file contains the Agent class, which is responsible for interacting with the environment, 
store experiences in memory and train the actor networks and critic network.
- **model.py:** This file contains the actor and critic network architecture used by the agent.
- **replay_buffer.py:** This file contains the ReplayBuffer class, which stores the experiences. 
The ReplayBuffer class is modified so that it can be used in a multi-agent environment.
- **run_trained.py:** This file can run a trained agent on the visual Tennis environment.
- **log/tensorboard:** This folder contains the tensorboard graph of a successful training run.
- **checkpoints:** This folder contains the states of actor networks and critic network of a successful training run.
<br/>

Every RL project should have well-defined state, action and reward spaces. For this project, the state, action and reward spaces are described below:  
- **State-space:** The Tennis environment is a world created using Unity. The environment consists of two moveable rackets, and a ball. 
State-space is an array representation of the environment consisting of 24 floating-point values.  
- **Action-space:** The action space consists of 2 continuous variables for each agent. 
The actions determine the movement of the racket and both are in the range -1 to 1.
- **Reward-space:** Rewards are separate for each racket agent. Each time a racket is able to send the ball over the net, it gets +0.1 point. 
If a racket lets the ball drop, it receives -0.01 reward.
- **Agent's goal:** The goal of each agent is to send the ball over the net as many times as it can. 
In this project, the environment is considered solved, when an agent is capable of achieving 0.5 score on average for the last 100 episodes.
<br/>

## Getting Started
- The following python libraries are required to run the project: pytorch, numpy, tensorboardx and unityagents
- The Tennis environment folder is not included in this github project, 
but can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).
<br/>

## Instructions
#### To Train the Agent
To train the agent, all the files and folders mentioned in the **Project Files**, should be saved in a directory. 
Then the **sovler.py** file should be run using a python 3 interpreter. Two things to note while running the project for training:
- The **sovler.py** assumes that the Unity ML-Agents Tennis environment is in the same directory as itself. The location of the 
environment directory can be updated in line no 13 of the **sovler.py** file. 
- The RUN_NAME (line 10 of **sovler.py**) corresponds to a specific run, 
and creates a tensordboard graph and checkpoint file with the given value.
Different runs should have different RUN_NAME values.
  
#### To Run a Trained Agent
Trained agents (network states) are stored in the checkpoints folder, containing the names **Test1actor1.pth**, **Test1actor2.pth** and **Test1critic.pth**. 
To run a trained agent, update the RUN_NAME in the **run_trained.py** file (line 8) if necessary and run the **run_trained.py** file using a python 3 interpreter.
<br/>  
    
## Results
Please check the [report](https://github.com/fahimfss/ProjectTennis/blob/master/REPORT.md) file for the implementation and result details.
