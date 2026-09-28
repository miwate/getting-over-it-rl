"""Self-imitation: raise a policy's likelihood of the actions in demos of its own successes (scripts/demos.py), in the
policy head only (value head untouched). A move a policy succeeds at by luck, like the launch from the top of the
Construction Wall onto the girder (decided in its first third of a second), becomes one it makes on purpose. Repeat
demos.py and imitate.py to climb: 4% -> 10% -> 14% on held-out wall-top states over two rounds.

    python scripts/imitate.py runs/a/checkpoints/ppo_1_steps.zip runs/demos/crux.npz runs/b/checkpoints/ppo_2_steps.zip
"""
import argparse
import os
import pickle
import shutil

import numpy as np
import torch
from stable_baselines3 import PPO

from goi_env import GoiEnv
from train import demo_steps, vecnormalize_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("demos", help=".npz from demos.py")
    ap.add_argument("out", help="checkpoint to write (ppo_<n>_steps.zip; its normalization is copied next to it)")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    z = np.load(args.demos)
    with open(vecnormalize_path(args.checkpoint), "rb") as f:
        norm = pickle.load(f)
    game = GoiEnv(args.game_dir, stall_steps=300, stall_ends=False)  # the demos' observations, replayed
    X, A = [], []
    try:
        for j, reached in enumerate(z["reached"]):
            for t, obs, a in demo_steps(game, z, j):
                if t > reached:
                    break
                X.append(norm.normalize_obs(obs[None])[0])
                A.append(a)
    finally:
        game.close()
    X, A = torch.as_tensor(np.array(X)), torch.as_tensor(np.array(A))

    model = PPO.load(args.checkpoint, device="cpu")
    policy = model.policy
    opt = torch.optim.Adam([*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters(),
                            policy.log_std], lr=args.lr)
    for epoch in range(args.epochs):
        for idx in torch.randperm(len(X)).split(256):
            loss = -policy.get_distribution(X[idx]).log_prob(A[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mse = ((policy.get_distribution(X).distribution.mean - A) ** 2).mean().item()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    shutil.copy(vecnormalize_path(args.checkpoint), vecnormalize_path(args.out))
    print(f"{len(X)} (observation, action) pairs from {len(z['reached'])} demos; mean action to demo action MSE {mse:.3f}"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
