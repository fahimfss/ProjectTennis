import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ValueNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc(x)


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_actions),
            nn.Tanh()
        )

    def forward(self, state):
        return self.fc(state)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0:self.num_actions]

