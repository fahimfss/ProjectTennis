from unityagents import UnityEnvironment
from agent import TennisAgent
import torch
import numpy as np

# Location of the Tennis visual environment
RUN_ENV_PATH = "/home/fahim/Downloads/Tennis_Linux/Tennis.x86_64"
RUN_NAME = "Test1"

# initialize the environment
env = UnityEnvironment(file_name=RUN_ENV_PATH)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# getting environment info
env_info = env.reset(train_mode=False)[brain_name]
t_state = env_info.vector_observations[0]
num_agents = len(env_info.vector_observations)
state_size = len(t_state)
action_size = brain.vector_action_space_size


state = env_info.vector_observations

# create agent and load actors saved checkpoints
agent = TennisAgent(state_size, action_size, num_agents)
for i in range(num_agents):
    agent.actor[i].load_state_dict(torch.load('checkpoints/' + RUN_NAME + 'actor' + str(i + 1) + '.pth'))

# initialize the score
score = np.array([0.0 for i in range(num_agents)])

# run the agent in the environment
while True:
    action = agent.act(state)                   # select an action using the trained agent
    env_info = env.step(action)[brain_name]     # send the action to the environment
    state = env_info.vector_observations        # update state values
    reward = env_info.rewards                   # get the reward
    done = env_info.local_done                  # see if episode has finished
    score += np.array(reward)                   # update the score

    if np.any(done):  # exit loop when the episode is finished
        env.close()
        break

print('Agent Score: ', np.max(score))
