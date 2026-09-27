"""PPO policies for train.py. In their own module so checkpoints load from any script."""
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class BoundedMeanPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with tanh on the mean action. With a plain Gaussian clipped to [-1, 1], the mean drifts far
    outside the box (median |mean| 1.64 at 2M steps, 4.5 later): most actions clip to full strength, the cursor stays
    pinned at the edge of its reach, and the exploration noise never reaches the inside of the box."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())  # same Linear, so the optimizer still holds it


class ResidualMLP(nn.Module):
    """SimBa-style trunk (Lee et al., 2024): a linear embedding, then blocks x + W2(relu(W1(LayerNorm(x)))) with a 4x
    wider hidden layer, then a LayerNorm. Normalization and residual paths are what let bigger networks help in RL, where
    plain deep MLPs tend to train worse."""

    def __init__(self, n_in, width, blocks):
        super().__init__()
        self.embed = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))
            for _ in range(blocks))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = x + block(x)
        return self.norm(x)


class ResidualExtractor(nn.Module):
    """Separate ResidualMLP trunks for the actor and the critic, in SB3's mlp_extractor interface."""

    def __init__(self, n_in, width, blocks):
        super().__init__()
        self.policy_net, self.value_net = ResidualMLP(n_in, width, blocks), ResidualMLP(n_in, width, blocks)
        self.latent_dim_pi = self.latent_dim_vf = width

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class BigPolicy(BoundedMeanPolicy):
    """BoundedMeanPolicy with ResidualMLP trunks, width x blocks (512 x 2: ~8.5M weights, ~100x the 2x256 MLP)."""

    def __init__(self, *args, width=512, blocks=2, **kwargs):
        self.width, self.blocks = width, blocks
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ResidualExtractor(self.features_dim, self.width, self.blocks)
