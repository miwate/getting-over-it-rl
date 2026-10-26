"""TOI events per physics step under a random policy in the training env (start bank states), XLA solver.
Prints a histogram of events per world-step and of the per-step batch maximum (the event loop's trip count).

    JAX_PLATFORMS=cpu .venv/bin/python scripts/toi_events.py --envs 64 --actions 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--actions", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.0, help="action std (actions are clipped to [-1, 1])")
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from goi_jax.game import sim as SM
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    cfg = TrainConfig(name="toi-events", num_envs=args.envs)
    env = St.make_env(cfg)
    p = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(0), args.envs)
    obs, st = jax.vmap(env.reset, (0, None))(keys, p)

    def act(s, a):  # one action = K physics steps, returning per-step TOI events
        mouse = SM.clamp_magnitude(jnp.clip(a, -1, 1) * p.mouse, jnp.float32(100.0))  # env.MAX_INPUT

        def one(sim, _):
            sim, info = env.sim.physics_step(p.model, p.sim, sim, mouse, "w")
            return sim, info.toi_events
        sim, ev = jax.lax.scan(one, s.sim, None, length=env.k)
        return s._replace(sim=env.sim.end_action(sim)), ev
    f = jax.jit(jax.vmap(act, axis_name="w"))
    rng = np.random.default_rng(0)
    evs = []
    for t in range(args.actions):
        a = jnp.asarray(rng.normal(0, args.scale, (args.envs, 2)).astype(np.float32))
        st, ev = f(st, a)
        evs.append(np.asarray(ev))
    ev = np.concatenate(evs, 1)  # (envs, steps)
    print("events per world-step hist (0..9, 10+):", np.bincount(np.minimum(ev.ravel(), 10), minlength=11).tolist())
    print("batch max per step hist:", np.bincount(np.minimum(ev.max(0), 10), minlength=11).tolist())
    print("mean events per world-step", float(ev.mean()), "mean batch max", float(ev.max(0).mean()))
    print("DONE")


if __name__ == "__main__":
    main()
