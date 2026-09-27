"""A PPO policy whose mean action is bounded. In its own module so checkpoints load from any script."""
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class BoundedMeanPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with tanh on the mean action. With a plain Gaussian clipped to [-1, 1], the mean drifts far
    outside the box (median |mean| 1.64 at 2M steps, 4.5 later): most actions clip to full strength, the cursor stays
    pinned at the edge of its reach, and the exploration noise never reaches the inside of the box."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())  # same Linear, so the optimizer still holds it
