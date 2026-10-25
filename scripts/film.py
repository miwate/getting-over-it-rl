"""Film a climb path (climb.py output) in a fresh game, with the agent's mouse input drawn over the picture: a
translucent stick in the top-right corner, its recent strokes trailing. The replay is checked against the path's
recorded progress.

    python scripts/film.py runs/climbs/lead.npz --out runs/climbs/best-527.mp4
"""
import argparse
import os
import subprocess
import wave

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from goi_env import OBS_FIELDS, GoiEnv
from replay import replay

PROGRESS, WON = OBS_FIELDS.index("progress"), OBS_FIELDS.index("won")


def mouse(frame, path, i):
    """The frame with mouse input path[i] (and the 20 before it) drawn over its top-right corner, translucent. Only the
    widget's box is composited: the whole 1080p frame took 19 ms of a frame's 33."""
    h, w = frame.shape[:2]
    r, pad = h // 11, h // 50
    x0, y0, c = w - 2 * r - h // 25 - pad, h // 25 - pad, r + pad  # the box's corner, and the stick's centre in it
    out = frame.copy()
    box = out[y0:y0 + 2 * c, x0:x0 + 2 * c]
    picture = Image.fromarray(box).convert("RGBA")
    layer = Image.new("RGBA", picture.size)
    d = ImageDraw.Draw(layer)
    d.ellipse((c - r, c - r, c + r, c + r), fill=(0, 0, 0, 100), outline=(255, 255, 255, 150), width=2)
    trail = path[max(0, i - 20):i + 1]
    for k, (ax, ay) in enumerate(trail):
        a, dot = int(30 + 120 * (k + 1) / len(trail)), max(2, r // 25)
        d.ellipse((c + ax * r - dot, c - ay * r - dot, c + ax * r + dot, c - ay * r + dot), fill=(255, 255, 255, a))
    ax, ay = path[i]
    dot = max(5, r // 7)
    d.line((c, c, c + ax * r, c - ay * r), fill=(255, 255, 255, 220), width=max(3, r // 20))
    d.ellipse((c + ax * r - dot, c - ay * r - dot, c + ax * r + dot, c - ay * r + dot), fill=(255, 255, 255, 240),
              outline=(0, 0, 0, 160), width=2)
    box[:] = np.asarray(Image.alpha_composite(picture, layer).convert("RGB"))
    return out


def video(out, crf=16, game=None):
    """An mp4 writer for 30 fps frames, encoded on the GPU; with the game the frames come from, with its sound."""
    if game is not None:
        return SoundVideo(out, crf, game)
    return imageio.get_writer(out, fps=30, codec="h264_nvenc", quality=None, pixelformat="yuv420p",
                              macro_block_size=1,  # keep 1080 lines (the default stretches them to 1088)
                              ffmpeg_params=["-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-preset", "p5"])


class SoundVideo:
    """video() with the game's sound: the plugin renders the mix offline, one step's worth per frame (frame-exact
    however fast filming runs), and each frame brings the sound of the steps since the last (client.audio); it is
    muxed into the mp4 on close."""

    def __init__(self, out, crf, game):
        self.out, self.game, self.silent = out, game, out[:-4] + "-silent.mp4"
        self.writer, self.pcm, self.frames, self.rate, self.channels = video(self.silent, crf), [], 0, 0, 0
        game.client.audio()  # what played before filming

    def append_data(self, frame):
        self.writer.append_data(frame)
        self.frames += 1
        rate, channels, pcm = self.game.client.audio()
        if pcm.size:
            self.rate, self.channels = rate, channels
            self.pcm.append(pcm)

    def close(self):
        self.writer.close()
        if not self.frames:  # filming failed: its own error says why
            return
        if not self.pcm:
            os.replace(self.silent, self.out)
            return
        wav = self.out[:-4] + ".wav"
        with wave.open(wav, "wb") as w:
            w.setnchannels(self.channels)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            w.writeframes((np.clip(np.concatenate(self.pcm), -1, 1) * 32767).astype("<i2").tobytes())
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", self.silent, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", self.out], check=True)
        os.remove(self.silent)
        os.remove(wav)


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
    writer = video(out, crf, game)
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
