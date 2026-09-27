"""Carry a checkpoint trained on the 63-value observation (before commit 04caf4c) over to today's 105-value one without
changing what it does: its weights for the new inputs (17 more route points ahead, the last 4 actions) start at zero, and
its normalization stats map over. The new inputs' stats come from a checkpoint trained on today's observation
(--stats-from), so they are normalized sensibly once training starts to use them.

    python scripts/port.py runs/rw-control/checkpoints/ppo_1500160_steps.zip runs/rw-port/checkpoints/ppo_1500160_steps.zip
        --stats-from runs/sa-small/checkpoints/ppo_6000640_steps.zip
"""
import argparse
import os
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO

from goi_env import OBS_FIELDS
from train import vecnormalize_path

OLD_FIELDS = ("pot_x", "pot_y", "pot_vx", "pot_vy", "pot_angle", "pot_angvel", "hammer_cos", "hammer_sin",
              "hammer_angvel", "hammer_ext", "hammer_extvel", "head_contact", "pot_contact", "fell_in_lake", "won",
              "progress", "ahead2_dx", "ahead2_dy", "ahead5_dx", "ahead5_dy", "ahead10_dx", "ahead10_dy", "cursor_dx",
              "cursor_dy", "cursor_head_dx", "cursor_head_dy", "mouse_x", "mouse_y", "mouse_speed")
N_RAYS = 32  # then the old observation had stuck and gap; the new one has the last 4 actions (8 values), stuck and gap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", help="checkpoint .zip trained on the 63-value observation")
    ap.add_argument("out", help="where to write the ported checkpoint (ppo_<steps>_steps.zip, for vecnormalize_path)")
    ap.add_argument("--stats-from", required=True, help="a checkpoint trained on today's observation")
    args = ap.parse_args()

    with open(vecnormalize_path(args.old), "rb") as f:
        norm = pickle.load(f)
    with open(vecnormalize_path(args.stats_from), "rb") as f:
        donor = pickle.load(f)
    n = donor.observation_space.shape[0]
    assert n == len(OBS_FIELDS) + N_RAYS + 8 + 2, "the donor's observation isn't today's"
    rays = range(len(OBS_FIELDS), len(OBS_FIELDS) + N_RAYS)
    where = np.array([OBS_FIELDS.index(f) for f in OLD_FIELDS] + list(rays) + [n - 2, n - 1])  # old input i -> new where[i]

    model = PPO.load(args.old, device="cpu")
    assert model.observation_space.shape == (len(where),), "not a 63-value checkpoint"
    policy = model.policy
    for net in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
        layer = torch.nn.Linear(n, net[0].out_features)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:, torch.as_tensor(where)] = net[0].weight
            layer.bias.copy_(net[0].bias)
        net[0] = layer
    policy.optimizer = policy.optimizer_class(policy.parameters(), lr=policy.optimizer.param_groups[0]["lr"],
                                              **policy.optimizer_kwargs)  # fresh: the old one's state has old shapes
    model.observation_space = donor.observation_space
    model._last_obs = None  # an old-shaped observation from its last rollout
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)

    mean, var = donor.obs_rms.mean.copy(), donor.obs_rms.var.copy()
    mean[where], var[where] = norm.obs_rms.mean, norm.obs_rms.var
    norm.obs_rms.mean, norm.obs_rms.var = mean, var  # count stays the old one's: the old inputs' stats barely move
    norm.observation_space = donor.observation_space
    norm.class_attributes = norm.returns = None  # VecNormalize's pickling drops these, and expects to find them
    with open(vecnormalize_path(args.out), "wb") as f:
        pickle.dump(norm, f)
    print(f"wrote {args.out}: {len(where)} inputs mapped, {n - len(where)} new at zero weight")


if __name__ == "__main__":
    main()
