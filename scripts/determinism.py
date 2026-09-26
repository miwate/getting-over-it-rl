"""Save a state, replay the same actions twice from it, and compare the two trajectories."""
import argparse
import os

import numpy as np

from goi_env import OBS_FIELDS, GoiEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--steps", type=int, default=200, help="actions per replay")
    ap.add_argument("--warmup", type=int, default=300, help="random actions before saving, to save mid-motion")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    env = GoiEnv(args.game_dir)
    rng = np.random.default_rng(0)
    env.reset()
    for _ in range(args.warmup):
        env.step(rng.uniform(-1, 1, 2))
    state_id = env.save_state()
    actions = rng.uniform(-1, 1, (args.steps, 2))

    def replay():
        env.load_state(state_id)
        return np.array([env.step(a)[0] for a in actions])

    a, b = replay(), replay()
    env.close()

    same = (a == b).all(axis=1)
    print(f"{args.steps} actions replayed twice from one saved state: {same.sum()}/{args.steps} steps bit-identical")
    if not same.all():
        t = int(np.argmin(same))
        worst = np.abs(a - b).max(axis=0)[:len(OBS_FIELDS)]
        print(f"first difference at step {t}; largest per field:", dict(zip(OBS_FIELDS, worst.round(6))))


if __name__ == "__main__":
    main()
