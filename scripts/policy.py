"""Policies for train.py (PPO) and rlpd.py (SAC). In their own module so checkpoints load from any script."""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
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


class Residual(nn.Module):
    """x + scale * W2(relu(W1(LayerNorm(x)))), W1 4x wider as in ResidualMLP, with a per-channel scale that starts at 0
    (LayerScale, Touvron et al., 2021): the block adds nothing until trained, then phases in. A zero W2 instead also adds
    nothing, but Adam's first steps move all of W2 by the learning rate in the same direction (its inputs are all >= 0):
    on rw-control that moved the policy 30x past the KL cap in one step."""

    def __init__(self, width):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))
        self.scale = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return x + self.scale * self.f(x)


class GrownPolicy(BoundedMeanPolicy):
    """BoundedMeanPolicy whose actor and critic trunks end in `grow` Residual blocks, which add nothing until trained,
    so a warm start from a 2x256 checkpoint (train.py --grow) behaves exactly as before, with more room to learn."""

    def __init__(self, *args, grow=2, **kwargs):
        self.grow = grow
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        width = self.mlp_extractor.latent_dim_pi
        for net in (self.mlp_extractor.policy_net, self.mlp_extractor.value_net):
            for _ in range(self.grow):
                net.append(Residual(width))


class BigPolicy(BoundedMeanPolicy):
    """BoundedMeanPolicy with ResidualMLP trunks, width x blocks (512 x 2: ~8.5M weights, ~100x the 2x256 MLP)."""

    def __init__(self, *args, width=512, blocks=2, **kwargs):
        self.width, self.blocks = width, blocks
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ResidualExtractor(self.features_dim, self.width, self.blocks)


if __name__ == "__main__":  # self-check: grown from a 2x256 policy, a GrownPolicy acts and values exactly as it does
    import gymnasium as gym
    obs, act = gym.spaces.Box(-1, 1, (105,)), gym.spaces.Box(-1, 1, (2,))
    kw = dict(net_arch=[256, 256], use_sde=True, log_std_init=-2)
    plain, grown = BoundedMeanPolicy(obs, act, lambda _: 1e-4, **kw), GrownPolicy(obs, act, lambda _: 1e-4, grow=2, **kw)
    grown.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(64, 105)
    with torch.no_grad():
        assert torch.equal(plain.get_distribution(x).distribution.mean, grown.get_distribution(x).distribution.mean)
        assert torch.equal(plain.predict_values(x), grown.predict_values(x))
    print(f"ok: {sum(p.numel() for p in plain.parameters()):,} -> {sum(p.numel() for p in grown.parameters()):,} weights")


class LayerNormSACPolicy(SACPolicy):
    """SAC's policy with LayerNorm after each hidden layer of every Q-network (RLPD, Ball et al., 2023: it keeps the
    critics from overestimating actions far from the data when half of each batch is offline transitions)."""

    def make_critic(self, features_extractor=None):
        critic = super().make_critic(features_extractor)
        for i, q in enumerate(critic.q_networks):
            layers = []
            for m in q:
                layers.append(m)
                if isinstance(m, nn.Linear) and m is not q[-1]:
                    layers.append(nn.LayerNorm(m.out_features))
            critic.q_networks[i] = nn.Sequential(*layers)
            critic.add_module(f"qf{i}", critic.q_networks[i])
        return critic
