"""Film a climb path (climb.py output) in a fresh game, with a side panel: the reward as training counts it (+1 per unit
of new route record, -0.001 per step), the agent's mouse input, and which search found each moment: the planner (the
networks' moves plus a nudge, the best of 16 two-second futures, re-planned every half second) or a Go-Explore segment
(random mouse strokes from an archive of saved states). The replay is checked against the path's recorded progress.

    python scripts/film.py runs/climbs/lead.npz --out runs/climbs/best-527-hud.mp4
"""
import argparse
import colorsys
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from goi_env import OBS_FIELDS, GoiEnv
from replay import replay

PROGRESS, WON = OBS_FIELDS.index("progress"), OBS_FIELDS.index("won")
PANEL = 640
BG, DIM, TEXT, GOOD = (18, 20, 26), (90, 96, 110), (225, 228, 235), (110, 220, 140)
PLAN, EXPLORE = (90, 160, 255), (255, 160, 60)
FONTS = "C:/Windows/Fonts/"


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


def modes(cuts, n, commit=15):
    """Per step: 0 inside a planner decision (cuts every `commit` steps), 1 inside a Go-Explore segment (any other
    length); and the index of the segment it's in."""
    # ponytail: a Go-Explore segment exactly `commit` steps long reads as the planner; rare, and climb.py doesn't log it
    edges = sorted(set(int(c) for c in cuts)) + [n]
    mode, seg = np.zeros(n, int), np.zeros(n, int)
    for k, (a, b) in enumerate(zip(edges, edges[1:])):
        mode[a:b] = 0 if b - a == commit or b == n and b - a < commit else 1
        seg[a:b] = k
    return mode, seg


def film(game, z, out, title, chain_file=None, height=1440, crf=16):
    """Film path z (a climb.py or marathon.py .npz) in `game`, which renders: replayed from the start, with the side
    panel. Prints whether the replay matched the recorded progress."""
    path, trace, cuts = z["try0"], z["trace"], set(int(c) for c in z["cuts0"])
    n = len(path)
    mode, seg = modes(cuts, n)
    if chain_file:  # which network of the chain plays at each step, by the progress it acts from
        chain = json.load(open(chain_file))
        names = [os.path.normpath(spec).split(os.sep)[1] for _, spec in chain]
        before = np.concatenate([[trace[0]], trace[:-1]])
        net = np.array([max(k for k, (start, _) in enumerate(chain) if p >= start or k == 0) for p in before])
        hues = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(k / len(chain), 0.55, 0.95)) for k in range(len(chain))]
    obs, _ = game.reset()
    best, rewards = float(obs[PROGRESS]), np.zeros(n)
    for t in range(n):  # what GoiEnv.step pays, from the recorded progress
        rewards[t] = max(0.0, trace[t] - best) - game.time_penalty
        best = max(best, trace[t])
    total = np.concatenate([[0.0], np.cumsum(rewards)])

    H, W = height, PANEL
    big, mid, small, mono = font("segoeuib.ttf", 64), font("segoeuib.ttf", 34), font("segoeui.ttf", 26), \
        font("consola.ttf", 28)
    gx, gy, gw, gh = 40, 330, W - 80, 170   # reward graph
    sx, sy, sr = W // 2, 780, 130           # mouse stick
    tx, ty, tw = 40, 1300, W - 80           # search timeline
    seconds = n / 30

    static = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(static)
    size = 34
    while size > 16 and d.textlength(title, font=font("segoeuib.ttf", size)) > W - 80:  # fit the panel
        size -= 1
    d.text((40, 30), title, font=font("segoeuib.ttf", size), fill=TEXT)
    d.line((40, 90, W - 40, 90), fill=DIM, width=2)
    d.text((40, 110), "REWARD", font=mid, fill=DIM)
    d.text((40, 270), "+1 per unit of new route record, -0.001 per step", font=small, fill=DIM)
    curve = [(gx + gw * t / n, gy + gh - gh * total[t] / max(total.max(), 1e-6)) for t in range(0, n + 1, 10)]
    d.line(curve, fill=DIM, width=2)
    d.text((40, 560), "MOUSE INPUT", font=mid, fill=DIM)
    d.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), outline=DIM, width=3)
    d.line((sx - sr, sy, sx + sr, sy), fill=DIM)
    d.line((sx, sy - sr, sx, sy + sr), fill=DIM)
    d.text((40, 1000), "PLAYED BY" if chain_file else "SEARCH", font=mid, fill=DIM)
    for t in range(n):  # the whole run, coloured by who chose the moves
        colour = hues[net[t]] if chain_file else EXPLORE if mode[t] else (40, 70, 120)
        d.rectangle((tx + tw * t / n, ty, tx + tw * (t + 1) / n, ty + 40), fill=colour)
    d.text((tx, ty + 50), "0:00", font=small, fill=DIM)
    d.text((tx + tw - 60, ty + 50), f"{int(seconds // 60)}:{int(seconds % 60):02d}", font=small, fill=DIM)

    writer = imageio.get_writer(out, fps=30, codec="libx264", quality=None, pixelformat="yuv420p",
                                ffmpeg_params=["-crf", str(crf), "-preset", "medium"])
    parted, planned, explored, last = None, 0, 0, -1
    try:
        for t in range(n + 1):
            if t < n and t in cuts:
                s = game.save_state()
                obs = game.load_state(s)
                game.drop_state(s)
            if t > 0 and parted is None and abs(obs[PROGRESS] - trace[t - 1]) > 1e-3:
                parted = t
            i = min(t, n - 1)
            if seg[i] != last:
                last = seg[i]
                planned, explored = planned + (mode[i] == 0), explored + (mode[i] == 1)
            panel = static.copy()
            d = ImageDraw.Draw(panel)
            r = rewards[t - 1] if t else 0.0
            d.text((40, 150), f"{total[t]:+.1f}", font=big, fill=TEXT)
            d.text((330, 170), f"{r:+.3f} this step", font=mono, fill=GOOD if r > 0 else DIM)
            d.line([p for p in curve if p[0] <= gx + gw * t / n] or [curve[0]], fill=GOOD, width=4)
            x = gx + gw * t / n
            d.ellipse((x - 7, gy + gh - gh * total[t] / max(total.max(), 1e-6) - 7, x + 7,
                       gy + gh - gh * total[t] / max(total.max(), 1e-6) + 7), fill=GOOD)
            trail = path[max(0, i - 20):i + 1]
            for k, (ax, ay) in enumerate(trail):
                c = int(60 + 150 * (k + 1) / len(trail))
                d.ellipse((sx + ax * sr - 5, sy - ay * sr - 5, sx + ax * sr + 5, sy - ay * sr + 5), fill=(c, c, c))
            ax, ay = path[i]
            d.line((sx, sy, sx + ax * sr, sy - ay * sr), fill=TEXT, width=4)
            d.ellipse((sx + ax * sr - 14, sy - ay * sr - 14, sx + ax * sr + 14, sy - ay * sr + 14), fill=TEXT)
            d.text((40, 940), f"x {ax:+.2f}   y {ay:+.2f}", font=mono, fill=TEXT)
            if chain_file:
                colour, label, what = hues[net[i]], names[net[i]], "network alone: no save/load, no search"
            else:
                colour, label, what = (EXPLORE, "GO-EXPLORE", "random strokes from saved states") if mode[i] else \
                    (PLAN, "PLANNER", "network + nudge, best of 16 futures")
            d.rounded_rectangle((40, 1060, W - 40, 1150), radius=16, fill=colour)
            d.text((64, 1072), label, font=mid, fill=BG)
            d.text((64, 1112), what, font=small, fill=BG)
            if chain_file:
                d.text((40, 1175), f"progress {obs[PROGRESS]:6.1f} / 864", font=mono, fill=TEXT)
                d.text((40, 1215), f"best     {trace[:max(t, 1)].max():6.1f}", font=mono, fill=GOOD)
            else:
                d.text((40, 1175), f"planner decisions  {planned}", font=mono, fill=PLAN)
                d.text((40, 1215), f"Go-Explore segments {explored}", font=mono, fill=EXPLORE)
            x = tx + tw * t / n
            d.line((x, ty - 8, x, ty + 48), fill=TEXT, width=4)
            clock = t / 30
            d.text((W // 2 - 40, ty + 50), f"{int(clock // 60)}:{int(clock % 60):02d}", font=small, fill=TEXT)
            writer.append_data(np.concatenate([game.render(), np.asarray(panel)], axis=1))
            if t < n:
                obs, *_ = game.step(path[t])
    finally:
        writer.close()
        game.close()
    print(f"wrote {out}: {n / 30:.0f} s, reward {total[-1]:+.1f}, "
          + ("same path as recorded" if parted is None else f"parted from the record at {parted / 30:.1f} s")
          + (", won" if obs[WON] > 0.5 else ", not won"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a climb.py .npz")
    ap.add_argument("--out", required=True, help=".mp4 to write")
    ap.add_argument("--title", default="Search agent, real game, real time")
    ap.add_argument("--width", type=int, default=2560, help="game picture width (the panel adds 640)")
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--crf", type=int, default=16, help="x264 quality: lower is better and bigger")
    ap.add_argument("--chain", help="a network chain (runs/chain.json) that played the path (marathon.py): the panel "
                                    "shows which network plays instead of search events")
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
    film(game, z, args.out, args.title, args.chain, args.height, args.crf)


if __name__ == "__main__":
    main()
