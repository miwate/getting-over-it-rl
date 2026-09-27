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


class Residual(nn.Module):
    """x + W2(relu(W1(LayerNorm(x)))), W1 4x wider, as in ResidualMLP."""

    def __init__(self, width):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))

    def forward(self, x):
        return x + self.f(x)


class GrownPolicy(BoundedMeanPolicy):
    """BoundedMeanPolicy whose actor and critic trunks end in `grow` Residual blocks with a zero last layer, which add
    nothing until training changes them, so a warm start from a 2x256 checkpoint (train.py --grow) behaves exactly as
    before, with more room to learn."""

    def __init__(self, *args, grow=2, **kwargs):
        self.grow = grow
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        width = self.mlp_extractor.latent_dim_pi
        for net in (self.mlp_extractor.policy_net, self.mlp_extractor.value_net):
            for _ in range(self.grow):
                net.append(Residual(width))

    def _build(self, lr_schedule):
        super()._build(lr_schedule)  # its weight initialization would overwrite a zero set any earlier
        for net in (self.mlp_extractor.policy_net, self.mlp_extractor.value_net):
            for block in net[-self.grow:] if self.grow else []:
                nn.init.zeros_(block.f[-1].weight)
                nn.init.zeros_(block.f[-1].bias)


class BigPolicy(BoundedMeanPolicy):
    """BoundedMeanPolicy with ResidualMLP trunks, width x blocks (512 x 2: ~8.5M weights, ~100x the 2x256 MLP)."""

    def __init__(self, *args, width=512, blocks=2, **kwargs):
        self.width, self.blocks = width, blocks
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ResidualExtractor(self.features_dim, self.width, self.blocks)


if __name__ == "__main__":  # self-check: grown from a 2x256 policy, a GrownPolicy acts and values exactly as it does
    import gymnasium as gym
    import torch
    obs, act = gym.spaces.Box(-1, 1, (105,)), gym.spaces.Box(-1, 1, (2,))
    kw = dict(net_arch=[256, 256], use_sde=True, log_std_init=-2)
    plain, grown = BoundedMeanPolicy(obs, act, lambda _: 1e-4, **kw), GrownPolicy(obs, act, lambda _: 1e-4, grow=2, **kw)
    grown.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(64, 105)
    with torch.no_grad():
        assert torch.equal(plain.get_distribution(x).distribution.mean, grown.get_distribution(x).distribution.mean)
        assert torch.equal(plain.predict_values(x), grown.predict_values(x))
    print(f"ok: {sum(p.numel() for p in plain.parameters()):,} -> {sum(p.numel() for p in grown.parameters()):,} weights")
