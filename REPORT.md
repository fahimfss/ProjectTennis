# Report on Project Tennis

## Overview

In this project, I used the MADDPG algorithm for solving the Unity Tennis environment. Here's a very basic overview of the training process:  

- Initially, the Unity Tennis environment is initialized. This environment is responsible for providing the state, reward, next-state and done 
(if an episode is completed) values.
- Then, an agent object is created. 
In MADDPG, the agent contains a critic network which outputs the state-value using combined state and action values of all the agents. 
The agent also contains multiple actor networks, which is two for the Tennis environment. Each actor network represent a racket agent
and outputs the action-values for a given state. In this project, 
the agent codes are written in the Agent class [(agent.py)](https://github.com/fahimfss/ProjectTennis/blob/master/agent.py) 
and the actor and critic network codes in ActorNetwork and CriticNetwork classes [(model.py)](https://github.com/fahimfss/ProjectTennis/blob/master/model.py)

- The agent picks an action based on the current state using the actor networks. 
Based on the action, the environment provides next-state, reward, and done values. This process is repeated for a very long time. 
- To choose better actions, the agent needs to learn by using the values provided by the environment. 
Instead of learning directly from the environment outputs (called **experience**), the agent stores those experiences in a buffer 
called the replay buffer and samples experiences from the buffer regularly for the learning purpose. The agent uses an object of the class ReplayBuffer 
[(replay_buffer.py)](https://github.com/fahimfss/ProjectTennis/blob/master/replay_buffer.py) for storing experiences.

- For learning, the agent picks sample experiences from the replay buffer. Then based on the MADDPG paper, the loss for the actor networks and
the critic network are calculated. Then back-propagation is applied on the loss values and the networks are trained. 

- After the training reaches a certain level (in this environment, when the mean reward reaches the value 1.0 for a single agent,
for the last 100 episodes), the training is finished.

### Hyperparameters
Following hyperparameters are defined in the **[agent.py](https://github.com/fahimfss/ProjectTennis/blob/master/agent.py)** file:    
ACTOR_LEARNING_RATE = 8e-4        
CRITIC_LEARNING_RATE = 3e-4        
BUFFER_SIZE = 1000000             
BATCH_SIZE = 128                    
GAMMA = 0.99                       
REPLAY_START = BATCH_SIZE * 4       
UPDATE_EVERY = 5                      
UPDATE_TIMES = 10                    
TAU = 1e-3  
   
### Network Architecture
The project contains two kinds of networks: ActorNetwork and CriticNetwork.  
Here's the CriticNetwork:  
```
CriticNetwork(
  (fcs1): Linear(in_features=48, out_features=512, bias=True)  
  (fc2): Linear(in_features=516, out_features=256, bias=True)  
  (fc3): Linear(in_features=256, out_features=128, bias=True)  
  (fc4): Linear(in_features=128, out_features=1, bias=True)  
)
```
Here's the ActorNetwork:
```
ActorNetwork(
  (fc1): Linear(in_features=24, out_features=384, bias=True)
  (fc2): Linear(in_features=384, out_features=2, bias=True)
)
```

## Results
The code in its current state was able to achieve a mean score of 1.0 over 100 episodes. 
The following table contains the summary of the train run:  
|Run Name|Episodes to reach mean rewards of 0.5|Episodes to reach mean rewards of 1.0|
|:-------|:----------------------------------:|:----------------------------------:|
|Test1|7651|7743|

Here's a plot of the mean reward over 100 episodes vs episode number:  
![TennisLog](https://user-images.githubusercontent.com/8725869/116815090-c2a83f80-ab7d-11eb-8bbb-319bef940aba.png)  
The mean rewads plot image can also be found [here](https://github.com/fahimfss/ProjectTennis/blob/master/log/tensorboard/Test1/tennis_rewards.png). The plot was created using tensorboard, with log files located at 
"[/log/tensorboard](https://github.com/fahimfss/ProjectTennis/tree/master/log/tensorboard/Test1)".    

Here's a video of [trained agents](https://github.com/fahimfss/ProjectTennis/tree/master/checkpoints) playing in the Tennis environment:  

https://user-images.githubusercontent.com/8725869/116815232-70b3e980-ab7e-11eb-9929-cd17b8783a94.mp4


## Future Works
- To solve the Soccer environment.  
- To solve the Tennis environment using other policy gradient algorithms like PPO, SAC.
