import torch
import torch.optim as optim
import numpy as np
from model import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer
from noise import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
ACTOR_LEARNING_RATE = 8e-4          # Learning rate of the actor (used in actor optimizers)
CRITIC_LEARNING_RATE = 3e-4         # Learning rate of the critic (used in critic optimizer)
BUFFER_SIZE = 1000000               # Size of the replay buffer
BATCH_SIZE = 128                    # Size of a single training batch of each agent
GAMMA = 0.99                        # Discount value
REPLAY_START = BATCH_SIZE * 4       # The agent will start learning after this many steps
UPDATE_EVERY = 5                    # The agent will learn every this many steps
UPDATE_TIMES = 10                   # In each learning process, the agent will use this many batches
TAU = 1e-3                          # Tau value used by the soft_update method


def soft_update(local_model, target_model, tau=TAU):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    :param local_model: (PyTorch model), weights will be copied from
    :param target_model: (PyTorch model), weights will be copied to
    :param tau: (float), interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class TennisAgent:
    """ The TennisAgent class is responsible for interacting with the tennis environment and
    teaching the neural network (model) to take appropriate actions based on states.

    This class uses the MADDPG architecture (paper link: https://arxiv.org/abs/1706.02275)
    """

    def __init__(self, num_inputs, num_outputs, num_multi_agents):
        """Initialize an ReacherAgent object.

        :param num_inputs: (int), dimension of each state
        :param num_outputs: (int), dimension of each action
        :param num_multi_agents: (int), number of agents in the env
        """

        # Creating critic, target-critic and critic optimizer. In MADDPG, the critic takes input the combined states
        # and combined actions of all agents and outputs the state-value.
        self.common_critic = CriticNetwork(num_inputs * num_multi_agents, num_outputs * num_multi_agents).to(device)
        self.target_common_critic = CriticNetwork(num_inputs * num_multi_agents,
                                                  num_outputs * num_multi_agents).to(device)
        self.common_critic_optimizer = optim.Adam(self.common_critic.parameters(),
                                                  lr=CRITIC_LEARNING_RATE, eps=1e-3)

        # Creating actor, target-actor and actor optimizer for each agent.
        # In MADDPG, the actor takes input the state and outputs action values corresponding to that state
        self.actor1 = ActorNetwork(num_inputs, num_outputs).to(device)
        self.target_actor1 = ActorNetwork(num_inputs, num_outputs).to(device)
        self.actor1_optimizer = optim.Adam(self.actor1.parameters(), lr=ACTOR_LEARNING_RATE, eps=1e-3)

        self.actor2 = ActorNetwork(num_inputs, num_outputs).to(device)
        self.target_actor2 = ActorNetwork(num_inputs, num_outputs).to(device)
        self.actor2_optimizer = optim.Adam(self.actor2.parameters(), lr=ACTOR_LEARNING_RATE, eps=1e-3)

        # Storing the actors, target-actors and actor optimizers in arrays
        self.actor = [self.actor1, self.actor2]
        self.target_actor = [self.target_actor1, self.target_actor2]
        self.actor_optimizer = [self.actor1_optimizer, self.actor2_optimizer]

        # Creating value criterion for the critic
        self.value_criterion = torch.nn.MSELoss()

        # Creating the replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, num_multi_agents)

        # t_step is used so that the learning takes place every "UPDATE_EVERY" steps
        self.t_step = 0

        # The number of agents in the env
        self.num_multi_agents = num_multi_agents

        # Creating the noise object
        self.noise = OUNoise(num_outputs, -1.0, 1.0)

    def step(self, state, action, reward, next_state, done):
        """Stores Experience in the replay buffer and calls the learn method in appropriate steps for training"""

        self.replay_buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.replay_buffer) >= REPLAY_START and self.t_step == 0:
            for i in range(UPDATE_TIMES):
                self.learn()

    def learn(self):
        """Collects batches of Experiences from the replay buffer and trains the actors and critic
        This method uses the update procedure described in the MADDPG paper.
        """

        sample = self.replay_buffer.sample(BATCH_SIZE)

        combined_state = None           # Combined current state values of each agent
        combined_action = None          # Combined action values of each agent
        combined_next_state = None      # Combined next state values of each agent
        combined_next_action = None     # Combined next action values of each agent, provided by target actors

        with torch.no_grad():
            for i in range(self.num_multi_agents):
                if i == 0:
                    combined_state = sample[i].states
                    combined_action = sample[i].actions
                    combined_next_state = sample[i].next_states
                    combined_next_action = self.target_actor[i](sample[i].next_states)

                else:
                    combined_state = torch.cat((combined_state, sample[i].states), 1)
                    combined_action = torch.cat((combined_action, sample[i].actions), 1)
                    combined_next_state = torch.cat((combined_next_state, sample[i].next_states), 1)
                    combined_next_action = torch.cat((combined_next_action,
                                                      self.target_actor[i](sample[i].next_states)), 1)

            # next state value is calculated using the critic network by passing
            # combined next states and combined next actions of each agent
            next_state_value = self.target_common_critic(combined_next_state, combined_next_action)

            # expected value is calculated for each agent (using rewards and discounted next state values)
            # and combined together
            expected_values = None
            for i in range(self.num_multi_agents):
                if i == 0:
                    expected_values = sample[i].rewards + (1.0 - sample[i].dones) * GAMMA * next_state_value
                else:
                    temp = sample[i].rewards + (1.0 - sample[i].dones) * GAMMA * next_state_value
                    expected_values = torch.cat((expected_values, temp), dim=1)

            # mean of the expected values are taken
            expected_values = expected_values.mean(dim=1).unsqueeze(1)

        # value is calculated using the critic network by passing
        # combined states and combined actions of each agent
        value = self.common_critic(combined_state, combined_action.detach())

        # value loss is calculated as the MSE between the value and the expected value
        value_loss = self.value_criterion(value, expected_values.detach())

        # the critic network is trained using the value loss
        self.common_critic_optimizer.zero_grad()
        value_loss.backward()
        self.common_critic_optimizer.step()

        # current state action value is calculated for each agent (using actor networks) and combined together
        combined_actor_action = None
        for i in range(self.num_multi_agents):
            if i == 0:
                combined_actor_action = self.actor[i](sample[i].states)
            else:
                temp = self.actor[i](sample[i].states)
                combined_actor_action = torch.cat((combined_actor_action, temp), dim=1)

        # policy loss is calculated by taking the negative mean of the output of the critic network,
        # which was generated using by the critic network using current combined state and combined action
        policy_loss = self.common_critic(combined_state, combined_actor_action)
        policy_loss = -policy_loss.mean()

        # all the actors are trained using policy loss value
        for i in range(self.num_multi_agents):
            self.actor_optimizer[i].zero_grad()
        policy_loss.backward()
        for i in range(self.num_multi_agents):
            self.actor_optimizer[i].step()

        # soft update is applied to the critic network and the actor networks
        soft_update(self.common_critic, self.target_common_critic)
        for i in range(self.num_multi_agents):
            soft_update(self.actor[i], self.target_actor[i])

    def act(self, states, add_noise=False, step=0):
        """This method uses the actor networks to generate action values for each agent
        :param states: (nd-array), state values of all agents.
        :param add_noise: (boolean), if true, noise is added to the action values
        :param step: the step parameter is used by the noise object to generate time-correlated noise
        :returns nd-array, containing action values for each agent
        """
        with torch.no_grad():
            ret = []
            for i in range(self.num_multi_agents):
                action = self.actor[i].get_action(states[i])
                if add_noise:
                    action = self.noise.get_action(action, step)
                ret.append(action)
            return np.array(ret)
