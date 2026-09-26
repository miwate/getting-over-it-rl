"""Random agent for N steps with rendering off and on, or with --instances, N games stepping in parallel."""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from goi_env import GoiEnv


def run(env, steps):
    env.reset()
    start = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    return steps / (time.perf_counter() - start)


def parallel(args):
    with ThreadPoolExecutor(args.instances) as pool:
        launches = [pool.submit(GoiEnv, args.game_dir, k=args.k) for _ in range(args.instances)]
    envs = [f.result() for f in launches if f.exception() is None]
    try:
        for f in launches:
            f.result()  # re-raise the first failed launch, once the games that did start are known and closable
        with ThreadPoolExecutor(args.instances) as pool:
            rates = list(pool.map(lambda env: run(env, args.steps), envs))
        print(f"{args.instances} games in parallel, rendering off: {sum(rates):6.0f} steps/s total "
              f"({min(rates):.0f}-{max(rates):.0f} per game)")
    finally:
        for env in envs:
            env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--k", type=int, default=4, help="physics steps (1/120 s) per action")
    ap.add_argument("--instances", type=int, default=0, help="instead, run this many games in parallel, rendering off")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    if args.instances:
        return parallel(args)
    for render_mode in (None, "human"):
        env = GoiEnv(args.game_dir, k=args.k, render_mode=render_mode, realtime=False)
        rate = run(env, args.steps)
        env.close()
        print(f"1 game, rendering {'on (uncapped)' if render_mode else 'off':13s}: {rate:6.0f} steps/s  "
              f"({rate * args.k:.0f} physics steps/s, {rate * args.k / 120:.1f}x real time)")


if __name__ == "__main__":
    main()
