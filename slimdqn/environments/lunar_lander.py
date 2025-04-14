import gymnasium as gym
import numpy as np
import jax


class LunarLander:
    def __init__(self, render_mode=None):
        self.env = gym.make("LunarLander-v3", render_mode=render_mode)
        self.observation_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.key = jax.random.PRNGKey(seed=0)
        self.random_action_probability = 0.2

    # Called when stored in the replay buffer
    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state)

    def reset(self):
        self.state, _ = self.env.reset()
        self.n_steps = 0

    def step(self, action):
        if self.random_action_probability > 0:
            action_key, self.key = jax.random.split(self.key)
            if jax.random.uniform(action_key) < self.random_action_probability:
                action = jax.random.randint(action_key, (), 0, self.n_actions).item()

        self.state, reward, absorbing, _, _ = self.env.step(action)
        self.n_steps += 1

        return reward, absorbing
