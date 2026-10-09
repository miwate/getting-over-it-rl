"""Export a climb path's states once (climb.py output: the search agent's winning paths), so training and tests can load
them anywhere along the route: replays of long paths drift in other launches (win-e4's parted at 301-567 s in some,
so its states above ~500 were silently missing from upper-section training). The path is replayed in fresh games until
one matches its recorded progress exactly; that replay's states are exported every --every steps, and at the first
arrival at each multiple of 5 (as marathon.py does): <path>-states.npz, loaded by train.py --demos and arrival_test.py.

    python scripts/path_states.py runs/climbs/win-e4.npz
"""
import argparse
import os

import numpy as np

from goi_env import OBS_FIELDS, GoiEnv
from train import demo_steps

PROGRESS = OBS_FIELDS.index("progress")
ARRIVAL, PERIODIC, MARK = 0, 1, 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a climb.py .npz (with its recorded progress, `trace`)")
    ap.add_argument("--every", type=int, default=10, help="steps between exported states")
    ap.add_argument("--launches", type=int, default=8)
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")
    z = np.load(args.path)
    trace = z["trace"]
    for launch in range(args.launches):
        game = GoiEnv(args.game_dir, stall_steps=300, stall_ends=False)
        states, mark, parted = [], MARK, None
        try:
            for t, obs, a in demo_steps(game, z, 0):
                p = float(obs[PROGRESS])
                if t > 0 and abs(p - trace[t - 1]) > 1e-3:
                    parted = t
                    break
                kinds = []
                while p >= mark:
                    kinds, mark = [ARRIVAL], mark + MARK
                if t % args.every == 0:
                    kinds.append(PERIODIC)
                for kind in kinds[:1]:
                    sid = game.save_state()
                    states.append((p, t, kind, game.export_state(sid)))
                    game.drop_state(sid)
        finally:
            game.close()
        if parted is None:
            break
        print(f"launch {launch + 1}: the replay parted from the record at {parted / 30:.1f} s; trying another")
    else:
        raise SystemExit(f"no exact replay in {args.launches} launches")
    out = args.path[:-4] + "-states.npz"
    np.savez(out, exported=1, progress=np.array([s[0] for s in states], np.float32),
             step=np.array([s[1] for s in states]), kind=np.array([s[2] for s in states]),
             **{f"s{i}": np.frombuffer(s[3], np.uint8) for i, s in enumerate(states)})
    print(f"wrote {out}: {len(states)} states, progress {states[0][0]:.0f} to {states[-1][0]:.0f}")


if __name__ == "__main__":
    main()
