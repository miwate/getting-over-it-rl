"""Film a climb path (climb.py output) in a fresh game, with the agent's mouse input drawn over the picture: a
translucent stick in the top-right corner, its recent strokes trailing. The replay is checked against the path's
recorded progress.

    python scripts/film.py runs/climbs/lead.npz --out runs/climbs/best-527.mp4
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from goi_env import OBS_FIELDS, GoiEnv
from replay import replay

PROGRESS, WON = OBS_FIELDS.index("progress"), OBS_FIELDS.index("won")


def mouse(frame, path, i):
    """The frame with mouse input path[i] (and the 20 before it) drawn over its top-right corner, translucent."""
    h, w = frame.shape[:2]
    r = h // 11
    cx, cy = w - r - h // 25, r + h // 25
    picture = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", picture.size)
    d = ImageDraw.Draw(layer)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 100), outline=(255, 255, 255, 150), width=2)
    trail = path[max(0, i - 20):i + 1]
    for k, (ax, ay) in enumerate(trail):
        a, dot = int(30 + 120 * (k + 1) / len(trail)), max(2, r // 25)
        d.ellipse((cx + ax * r - dot, cy - ay * r - dot, cx + ax * r + dot, cy - ay * r + dot), fill=(255, 255, 255, a))
    ax, ay = path[i]
    dot = max(5, r // 7)
    d.line((cx, cy, cx + ax * r, cy - ay * r), fill=(255, 255, 255, 220), width=max(3, r // 20))
    d.ellipse((cx + ax * r - dot, cy - ay * r - dot, cx + ax * r + dot, cy - ay * r + dot), fill=(255, 255, 255, 240),
              outline=(0, 0, 0, 160), width=2)
    return np.asarray(Image.alpha_composite(picture, layer).convert("RGB"))


def film(game, z, out, crf=16, start=None):
    """Film path z (a climb.py or marathon.py .npz) in `game`, which renders: replayed from the start (or from the
    exported state `start`), with the mouse input drawn over it. Prints whether the replay matched the recorded
    progress."""
    path, trace, cuts = z["try0"], z["trace"], set(int(c) for c in z["cuts0"])
    n = len(path)
    obs, _ = game.reset()
    if start is not None:
        sid = game.import_state(start)
        obs = game.load_state(sid)
        game.drop_state(sid)
    first = float(obs[PROGRESS])
    writer = imageio.get_writer(out, fps=30, codec="libx264", quality=None, pixelformat="yuv420p",
                                ffmpeg_params=["-crf", str(crf), "-preset", "medium"])
    parted = None
    try:
        for t in range(n + 1):
            if t < n and t in cuts:
                s = game.save_state()
                obs = game.load_state(s)
                game.drop_state(s)
            if t > 0 and parted is None and abs(obs[PROGRESS] - trace[t - 1]) > 1e-3:
                parted = t
            writer.append_data(mouse(game.render(), path, min(t, n - 1)))
            if t < n:
                obs, *_ = game.step(path[t])
    finally:
        writer.close()
        game.close()
    print(f"wrote {out}: {n / 30:.0f} s, progress {first:.1f} to {trace[-1]:.1f}, "
          + ("same path as recorded" if parted is None else f"parted from the record at {parted / 30:.1f} s")
          + (", won" if obs[WON] > 0.5 else ", not won"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a climb.py .npz")
    ap.add_argument("--out", required=True, help=".mp4 to write")
    ap.add_argument("--width", type=int, default=2560, help="game picture width")
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--crf", type=int, default=16, help="x264 quality: lower is better and bigger")
    ap.add_argument("--launches", type=int, default=6, help="game launches to try for an exact replay")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    args = ap.parse_args()
    if not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    z = np.load(args.path)
    for launch in range(args.launches):  # a launch that replays the path exactly (rarely one doesn't): film in it
        game = GoiEnv(args.game_dir, stall_steps=300, stall_ends=False, render_mode="rgb_array", width=args.width,
                      height=args.height)
        parted = replay(game, z)[1]
        if parted is None:
            break
        print(f"launch {launch + 1}: the replay parted from the record at {parted / 30:.1f} s; trying another")
        game.close()
    else:
        raise SystemExit(f"no exact replay in {args.launches} launches")
    film(game, z, args.out, args.crf)


if __name__ == "__main__":
    main()
