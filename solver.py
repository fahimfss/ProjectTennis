import torch
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment
from agent import TennisAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_NAME = "Test2"

# Initializing the Unity Banana environment
env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64", worker_id=1)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Initializing the state_size and action_size for the environment
t_env_info = env.reset(train_mode=True)[brain_name]
t_state = t_env_info.vector_observations[0]
num_agents = len(t_env_info.vector_observations)
state_size = len(t_state)
action_size = brain.vector_action_space_size
print('State shape: ', state_size)
print('Number of actions: ', action_size)
print('Number of agents: ', num_agents)

# Creating the TennisAgent object
agent = TennisAgent(state_size, action_size, num_agents)


def maddpq():
    """This method collects values from the environment and passes those to the agent for training.
    This method also saves training data in "log/tensorboard/" and trained agents in "checkpoints/"
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)
    writer = SummaryWriter(log_dir="log/tensorboard/" + RUN_NAME)  # initialize writer object for tensorboard

    i_episode = 0
    step = 0
    solved_episode = 0

    while True:
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.array([0.0 for i in range(num_agents)])

        while True:
            action = agent.act(state, add_noise=True, step=step)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            done_numpy = np.array(done).astype(np.float32)

            agent.step(state, action, reward, next_state, done_numpy)

            state = next_state
            score += np.array(reward)
            step += 1

            if np.any(done):
                break

        score = np.max(score)

        if len(scores_window) > 0:
            writer.add_scalar("score_mean_100", np.mean(scores_window), i_episode)
        writer.add_scalar("score", score, i_episode)
        writer.flush()

        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode > 0 and i_episode % 500 == 0:
            print()

        # Tf the mean score is 1.0, the training is finished
        if np.mean(scores_window) >= 1.0:
            if solved_episode == 0:
                solved_episode = i_episode

            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'
                  .format(i_episode, np.mean(scores_window)))

            print('\nReached 0.5 mean on episode {:d}'.format(solved_episode))

            torch.save(agent.common_critic.state_dict(), 'checkpoints/' + RUN_NAME + 'critic.pth')
            for i in range(num_agents):
                torch.save(agent.actor[i].state_dict(), 'checkpoints/' + RUN_NAME + 'actor' + str(i + 1) + '.pth')

            break

        if np.mean(scores_window) >= 0.5 and solved_episode == 0:
            solved_episode = i_episode

        i_episode += 1

    writer.close()
    env.close()


maddpq()
