import numpy as np


class OUNoise(object):
    """This class is responsible for adding time-correlated noise to the actions
    taken by the policy (actor). The class uses the Ornstein-Uhlenbeck process.

    Source: https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb
    """

    def __init__(self, action_size, low, high, mu=0.0, theta=0.15, max_sigma=0.12,
                 min_sigma=0.1, decay_period=50000):
        """Initialize parameters and noise process."""
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_size
        self.low = low
        self.high = high
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        """Update and return the internal state."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        """Apply noise to given action values and return it."""

        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)