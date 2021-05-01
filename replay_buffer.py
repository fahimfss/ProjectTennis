import random
import torch
import numpy as np
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
CombinedExperience = namedtuple('CombinedExperience', ['combined_state', 'combined_next_state', 'combined_action'])
Experiences = namedtuple('Experiences', ['states', 'actions', 'rewards', 'next_states', 'dones'])


class ReplayBuffer:
    def __init__(self, capacity, multi_agents):
        """Initialize parameters and build replay buffer.

        :param capacity: (int) Max size of replay buffer
        :param multi_agents: (int) Number of multiple agents in the environment
        """

        self.capacity = capacity
        self.multi_agents = multi_agents
        self.buffer = [[] for _ in range(multi_agents)]
        self.combined_buffer = []
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
            self.combined_buffer.append(None)

        for i in range(self.multi_agents):
            self.buffer[i][self.position] = Experience(state[i], action[i], reward[i], next_state[i], done[i])
        self.combined_buffer[self.position] = CombinedExperience(state.flatten(), next_state.flatten(), action.flatten())

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ Returns a batch of samples for each agents.

        :param batch_size : (int), The size of the batch

        :returns: (first) List of Experiences namedtuple. The size of the list equals to the number of agents.
                  Each Experiences contains a batch of states, actions, rewards, next_states and dones values.
                  (second) A dictionary containing a batch of combined state, nextL_state and action values.
        """

        assert len(self.buffer[0]) >= batch_size
        batch_indices = random.sample(range(0, len(self.buffer[0])), batch_size)
        ret = []

        combined_states = np.empty((batch_size, self.combined_buffer[0].combined_state.shape[0]))
        combined_next_states = np.empty((batch_size, self.combined_buffer[0].combined_next_state.shape[0]))
        combined_actions = np.empty((batch_size, self.combined_buffer[0].combined_action.shape[0]))

        for i, i_index in enumerate(batch_indices):
            combined_states[i] = self.combined_buffer[i_index].combined_state
            combined_next_states[i] = self.combined_buffer[i_index].combined_next_state
            combined_actions[i] = self.combined_buffer[i_index].combined_action

        combined_states = torch.FloatTensor(combined_states).to(device)
        combined_next_states = torch.FloatTensor(combined_next_states).to(device)
        combined_actions = torch.FloatTensor(combined_actions).to(device)

        ret_comb = {'combined_states': combined_states,
                    'combined_next_states': combined_next_states,
                    'combined_actions': combined_actions}

        for i in range(self.multi_agents):
            states = np.empty((batch_size, self.buffer[0][0].state.shape[-1]))
            actions = np.empty((batch_size, self.buffer[0][0].action.shape[-1]))
            rewards = np.empty((batch_size, 1))
            next_states = np.empty((batch_size, self.buffer[0][0].next_state.shape[-1]))
            dones = np.empty((batch_size, 1))
            for j, j_index in enumerate(batch_indices):
                states[j] = self.buffer[i][j_index].state
                actions[j] = self.buffer[i][j_index].action
                rewards[j] = self.buffer[i][j_index].reward
                next_states[j] = self.buffer[i][j_index].next_state
                dones[j] = self.buffer[i][j_index].done
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            ret.append(Experiences(states, actions, rewards, next_states, dones))

        return ret, ret_comb

    def __len__(self):
        return len(self.buffer[0])
