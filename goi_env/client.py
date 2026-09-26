"""Python side of the GOI RL Bridge. The wire protocol is documented in plugin/Server.cs.

Smoke test (stdlib only; launches the game, exercises every command, then quits it):
    python -m goi_env.client --game-dir "C:/.../steamapps/common/Getting Over It"
"""
import msvcrt
import os
import socket
import struct
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# Observation layout. The plugin appends n_rays distances from the pot to the terrain, counter-clockwise from +x.
OBS_FIELDS = (
    "pot_x", "pot_y", "pot_vx", "pot_vy", "pot_angle", "pot_angvel",  # world units, radians
    "hammer_cos", "hammer_sin", "hammer_angvel", "hammer_ext", "hammer_extvel",
    "head_contact", "pot_contact", "fell_in_lake", "won",  # 0/1 flags
)


def free_port():
    """A localhost TCP port nothing is listening on right now (other software may own fixed ones)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _one_game_starting():
    """Machine-wide lock: BepInEx rewrites its shared config file as each game starts, and games that start together
    come up without the mod. Windows releases the lock if its holder dies."""
    with open(os.path.join(tempfile.gettempdir(), "goi-rl-start.lock"), "a+b") as f:
        while True:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.2)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def start(game_dir, port=None, width=640, height=360, **config):
    """Launch the game in RL mode and connect to it; returns (process, client). Config as in GoiClient.configure.

    Games start one at a time, across all processes, until their level has loaded. A game quits when its client
    disconnects.
    """
    port = port or free_port()
    exe = Path(game_dir) / "GettingOverIt.exe"
    with _one_game_starting():
        proc = subprocess.Popen([str(exe), "-goi-rl", "-goi-port", str(port), "-screen-fullscreen", "0",
                                 "-screen-width", str(width), "-screen-height", str(height)], cwd=game_dir)
        try:
            return proc, GoiClient(port=port, width=width, height=height, **config)
        except BaseException:
            proc.kill()  # a game we launched but couldn't reach would never quit on its own
            raise


class GoiClient:
    """Lockstep connection to one game instance.

    Actions are hammer targets relative to the pot, in world units (clamped to the game's 3.5 reach).
    """

    def __init__(self, port=5555, timeout=180.0, **config):
        deadline = time.monotonic() + 60
        while True:  # the bridge starts listening a few seconds after launch
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError as e:
                if time.monotonic() > deadline:
                    raise ConnectionError(
                        f"no bridge on port {port} after 60 s. Is the game running with -goi-rl? With BepInEx's "
                        "default disk logging, at most 5 modded games can run from one install at a time.") from e
                time.sleep(0.5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)  # the first reply waits for the level to load
        self.configure(**config)
        self.sock.settimeout(30)

    def configure(self, k=4, n_rays=16, ray_length=10.0, render=False, width=0, height=0):
        """k physics steps (1/120 s each) per action; width/height 0 keeps the window size."""
        self.obs_size = self._int(b"k" + struct.pack("<iifiii", k, n_rays, ray_length, int(render), width, height))
        self._obs = struct.Struct(f"<{self.obs_size}f")
        return self.obs_size

    def reset(self):
        return self._obs_call(b"r")

    def step(self, dx, dy):
        return self._obs_call(b"s" + struct.pack("<ff", dx, dy))

    def save_state(self):
        return self._int(b"v")

    def load_state(self, state_id):
        obs = self._obs_call(b"l" + struct.pack("<i", state_id))
        if obs is None:
            raise KeyError(f"unknown state id {state_id}")
        return obs

    def frame(self):
        """The image rendered after the last step (needs render=True): width, height, RGB bytes with rows top-down."""
        data = self._call(b"f")
        width, height = struct.unpack_from("<ii", data)
        return width, height, data[8:]

    def close(self):
        """Quit the game and close the connection."""
        try:
            self._call(b"c")
        finally:
            self.sock.close()

    def _call(self, msg):
        self.sock.sendall(msg)
        (n,) = struct.unpack("<i", self._recv(4))
        return self._recv(n)

    def _recv(self, n):
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            r = self.sock.recv_into(view[got:])
            if not r:
                raise ConnectionError("the game closed the connection")
            got += r
        return buf

    def _obs_call(self, msg):
        data = self._call(msg)
        return self._obs.unpack(data) if data else None

    def _int(self, msg):
        return struct.unpack("<i", self._call(msg))[0]


def _smoke_test():
    import argparse
    import os
    import random

    ap = argparse.ArgumentParser(description="Launch the game in RL mode and exercise every bridge command.")
    ap.add_argument("--game-dir", default=os.environ.get("GOI_GAME_DIR"), help="install folder (or set GOI_GAME_DIR)")
    ap.add_argument("--port", type=int, help="default: a free port, or 5555 with --no-launch")
    ap.add_argument("--no-launch", action="store_true", help="connect to a game that is already running")
    args = ap.parse_args()
    if not args.no_launch and not args.game_dir:
        ap.error("pass --game-dir or set GOI_GAME_DIR")

    proc, c = (None, GoiClient(port=args.port or 5555)) if args.no_launch else start(args.game_dir, args.port)
    try:
        obs = c.reset()
        print("reset:", {f: round(v, 3) for f, v in zip(OBS_FIELDS, obs)})
        print("rays:", [round(r, 2) for r in obs[len(OBS_FIELDS):]])

        rng = random.Random(0)
        acts = [(rng.uniform(-3.5, 3.5), rng.uniform(-3.5, 3.5)) for _ in range(300)]
        t = time.perf_counter()
        for a in acts[:200]:
            obs = c.step(*a)
        print(f"200 random steps, rendering off: {200 / (time.perf_counter() - t):.0f} steps/s; "
              f"pot at ({obs[0]:.2f}, {obs[1]:.2f})")

        def worst(run1, run2):
            return max(abs(x - y) for a, b in zip(run1, run2) for x, y in zip(a, b))

        sid, saved = c.save_state(), obs
        original = [c.step(*a) for a in acts[200:]]
        print(f"load restores the saved observation: max diff {worst([saved], [c.load_state(sid)]):.3g}")
        replay1 = [c.step(*a) for a in acts[200:]]
        c.load_state(sid)
        replay2 = [c.step(*a) for a in acts[200:]]
        c.configure(render=True, width=640, height=360)
        c.load_state(sid)
        t = time.perf_counter()
        replay3 = [c.step(*a) for a in acts[200:]]
        rate = 100 / (time.perf_counter() - t)
        print(f"100 steps, replay 1 vs replay 2 from the same save: max diff {worst(replay1, replay2):.3g}")
        print(f"100 steps, replay with rendering on ({rate:.0f} steps/s) vs off: max diff {worst(replay1, replay3):.3g}")
        print(f"100 steps, uninterrupted original vs replay after load: max diff {worst(original, replay1):.3g} "
              "(a load rebuilds the physics world, so these need not match)")
        c.close()
        if proc:
            proc.wait(timeout=30)
    finally:
        if proc and proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    _smoke_test()
