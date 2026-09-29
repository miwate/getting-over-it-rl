"""Distill planner labels (plan.py's observations and actions) into a policy: regress its mean action onto the planner's
actions with squared error, in the policy head only. Not log-likelihood: maximizing it drives the policy's noise toward
zero and blows up just as it starts to fit (another RL project for this game measured log pi falling from -1 to -642
in 200 updates), and tests play the mean action plus a little noise anyway. Several label files aggregate (DAgger).

    python scripts/distill.py runs/x/checkpoints/ppo_1_steps.zip runs/plans/a.npz runs/plans/b.npz --out runs/y/checkpoints/ppo_2_steps.zip
"""
import argparse
import os
import pickle
import shutil

import numpy as np
import torch
from stable_baselines3 import PPO

from train import vecnormalize_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("labels", nargs="+", help=".npz files from plan.py")
    ap.add_argument("--out", required=True, help="checkpoint to write (ppo_<n>_steps.zip; its normalization is copied)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    X = np.concatenate([np.load(f)["obs"] for f in args.labels])
    A = np.concatenate([np.load(f)["actions"] for f in args.labels])
    with open(vecnormalize_path(args.checkpoint), "rb") as f:
        norm = pickle.load(f)
    X = torch.as_tensor(norm.normalize_obs(X))
    A = torch.as_tensor(A)
    held = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[: len(X) // 10]
    train = torch.ones(len(X), dtype=torch.bool)
    train[held] = False

    model = PPO.load(args.checkpoint, device="cpu")
    policy = model.policy
    mean = lambda x: policy.get_distribution(x).distribution.mean
    opt = torch.optim.Adam([*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters()], lr=args.lr)
    with torch.no_grad():
        before = ((mean(X[held]) - A[held]) ** 2).mean().item()
    Xt, At = X[train], A[train]
    for _ in range(args.epochs):
        for idx in torch.randperm(len(Xt)).split(256):
            loss = ((mean(Xt[idx]) - At[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        after = ((mean(X[held]) - A[held]) ** 2).mean().item()
        fit = ((mean(Xt) - At) ** 2).mean().item()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    shutil.copy(vecnormalize_path(args.checkpoint), vecnormalize_path(args.out))
    print(f"{len(X)} labels; held-out squared error {before:.3f} -> {after:.3f} (training {fit:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
