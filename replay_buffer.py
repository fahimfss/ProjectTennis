import random
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity, multi_agents):
        """Initialize parameters and build replay buffer.

        :param capacity: (int) Max size of replay buffer
        :param multi_agents: (int) Number of multiple agents in the environment
        """

        self.capacity = capacity
        self.multi_agents = multi_agents
        self.buffer = [[] for _ in range(multi_agents)]
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """ Add values to the replay buffer. The left most dimension of the parameters represent the agent.

        :param state: (nd-array), State values of the agents.
        :param action: (nd-array), Action values of the agents.
        :param reward: (float array), Reward values of the agents.
        :param next_state: (nd-array), Next state values of the agents.
        :param done: (float array), Done values of the agents.
        """
        if len(self.buffer[0]) < self.capacity:
            for i in range(self.multi_agents):
                self.buffer[i].append(None)

        for i in range(self.multi_agents):
            self.buffer[i][self.position] = (state[i], action[i], reward[i], next_state[i], done[i])

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ Returns a batch of samples for each agents.

        :param batch_size : (int), The size of the samples for the batch
        :returns: Array of tuples. Each tuple contains the state, action, reward,
                  next_state and done values for an agent.
        """

        assert len(self.buffer[0]) >= batch_size
        batch_indices = random.sample(range(0, len(self.buffer[0])), batch_size)
        ret = []

        for i in range(self.multi_agents):
            states = np.empty((batch_size, self.buffer[0][0][0].shape[-1]))
            actions = np.empty((batch_size, self.buffer[0][0][1].shape[-1]))
            rewards = np.empty((batch_size, 1))
            next_states = np.empty((batch_size, self.buffer[0][0][3].shape[-1]))
            dones = np.empty((batch_size, 1))
            for j in range(len(batch_indices)):
                states[j] = self.buffer[i][j][0]
                actions[j] = self.buffer[i][j][1]
                rewards[j] = self.buffer[i][j][2]
                next_states[j] = self.buffer[i][j][3]
                dones[j] = self.buffer[i][j][4]
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            ret.append((states, actions, rewards, next_states, dones))

        return ret

    def __len__(self):
        return len(self.buffer[0])
