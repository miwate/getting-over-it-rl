"""Experience collection for train.py's PPO without lockstep (train.py --collector async). SB3's SubprocVecEnv steps all
games together, so each step waits for the slowest: a game reloading a state (~20 ms, at every reset and rewind) held up
the others, and with the network's forward pass in the main process, training used ~1,700 steps/s where 10 games alone
step ~12,800/s (profile, 2026-10-06: 2.4 ms of each 3.6 ms step was waiting; PPO's update ~2% of the time).

Here each game has its own worker with a copy of the policy: per rollout it gets the current weights and normalization
statistics, plays its share of steps at its own pace, normalizes observations and rewards and bootstraps truncated
episodes as VecNormalize and PPO would, and sends the arrays back. The main process fills PPO's rollout buffer, updates
the statistics and runs PPO's own update, so checkpoints are the same as with lockstep collection. The statistics a
worker uses are those of the previous update (VecNormalize updates them every step): a negligible lag.
"""
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from stable_baselines3.common.utils import configure_logger, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


class Spaces(VecEnv):
    """A VecEnv with spaces only, so PPO and VecNormalize can be built in a process that runs no game."""

    def __init__(self, n, observation_space, action_space):
        super().__init__(n, observation_space, action_space)

    def reset(self):
        return np.zeros((self.num_envs, *self.observation_space.shape), np.float32)

    def step_async(self, actions):
        raise NotImplementedError("the collector steps the games")

    def step_wait(self):
        raise NotImplementedError("the collector steps the games")

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


GAME_LOST = (ConnectionError, OSError, EOFError)  # the game process died (rare: a space run lost one after 840k steps)


def _worker(conn, env_fn):
    torch.set_num_threads(1)  # its noise seed is random: runs differ, like lockstep ones
    env = env_fn.var()

    def fresh(error):
        """A new game in place of one that died: the run goes on, the episode is lost."""
        nonlocal env
        print(f"collector: game lost ({error!r}); starting a new one", flush=True)
        try:
            env.close()
        except Exception:
            pass
        env = env_fn.var()
        return env.reset()[0]
    conn.send((env.observation_space, env.action_space))
    policy = conn.recv().var()
    policy.set_training_mode(False)
    low, high = env.action_space.low, env.action_space.high
    seed = int(os.environ["GOI_COLLECTOR_SEED"]) if "GOI_COLLECTOR_SEED" in os.environ else None
    obs, _ = env.reset()
    ret = ep_reward = 0.0
    ep_length, start = 0, True
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            weights, (mean, var, clip_obs, eps), (ret_var, clip_reward, norm_reward), gamma, n, sde_freq = msg
            policy.load_state_dict(weights)

            def norm(o):
                return np.clip((o - mean) / np.sqrt(var + eps), -clip_obs, clip_obs).astype(np.float32)

            def value(o):
                with torch.no_grad():
                    return policy.predict_values(torch.as_tensor(norm(o)[None]))[0].item()

            O, A, V, L, R, S, raw, returns, episodes = [], [], [], [], [], [], [], [], []
            if seed is not None:  # tests: the same noise as a lockstep run seeded alike
                torch.manual_seed(seed)
                seed = None
            if policy.use_sde:  # as SB3: at the rollout's start, then every sde_freq steps from step 0
                policy.reset_noise(1)
            for t in range(n):
                if policy.use_sde and sde_freq > 0 and t % sde_freq == 0:
                    policy.reset_noise(1)
                while True:
                    o = norm(obs)
                    with torch.no_grad():
                        a, v, lp = policy(torch.as_tensor(o[None]))
                    a = a.numpy()[0]
                    try:
                        new_obs, r, terminated, truncated, info = env.step(np.clip(a, low, high))
                        break
                    except GAME_LOST as e:  # a new episode in a new game, from this step
                        obs, start, ret, ep_reward, ep_length = fresh(e), True, 0.0, 0.0, 0
                ret = ret * gamma + r
                raw.append(obs)
                returns.append(ret)
                rn = float(np.clip(r / np.sqrt(ret_var + eps), -clip_reward, clip_reward)) if norm_reward else float(r)
                if truncated and not terminated:  # a time limit or a rewind: the episode would have gone on
                    rn += gamma * value(new_obs)
                O.append(o), A.append(a), V.append(v.item()), L.append(lp.item()), R.append(rn), S.append(start)
                ep_reward += r
                ep_length += 1
                start = terminated or truncated
                if start:
                    episodes.append(dict(r=ep_reward, l=ep_length, **{k: info[k] for k in
                                                                       ("best_progress", "from_bottom", "rewinds")
                                                                       if k in info}))
                    ret = ep_reward = 0.0
                    ep_length = 0
                    try:
                        obs, _ = env.reset()
                    except GAME_LOST as e:
                        obs = fresh(e)
                else:
                    obs = new_obs
            conn.send((np.array(O), np.array(A), np.array(V, np.float32), np.array(L, np.float32),
                       np.array(R, np.float32), np.array(S, np.float32), np.array(raw, np.float32),
                       np.array(returns), value(obs), float(start), episodes))
    finally:
        env.close()


class Collector:
    """One worker process per game (env_fns: functions that build a game's env)."""

    def __init__(self, env_fns):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for fn in env_fns:
            here, there = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(there, CloudpickleWrapper(fn)), daemon=True)
            p.start()
            there.close()
            self.conns.append(here)
            self.procs.append(p)
        self.observation_space, self.action_space = [c.recv() for c in self.conns][0]

    def learn(self, model, norm, steps, save_every, checkpoints, save, reset_num_timesteps):
        """PPO.learn with this collection: `steps` more environment steps; save(path) writes a checkpoint (model and
        normalization) every `save_every` steps."""
        policy = model.policy
        cls, spaces, use_sde, kwargs = (model.policy_class, (model.observation_space, model.action_space),
                                        model.use_sde, dict(model.policy_kwargs))
        spec = CloudpickleWrapper(lambda: cls(*spaces, lambda _: 0.0, use_sde=use_sde, **kwargs))
        for c in self.conns:
            c.send(spec)
        if reset_num_timesteps:
            model.num_timesteps = 0
        total, start_steps, start_time = model.num_timesteps + steps, model.num_timesteps, time.time()
        model.set_logger(configure_logger(model.verbose, model.tensorboard_log, "tb", reset_num_timesteps))
        n, record, episodes = model.n_steps, -np.inf, []
        next_save = (model.num_timesteps // save_every + 1) * save_every
        buf = model.rollout_buffer
        while model.num_timesteps < total:
            weights = {k: v.cpu() for k, v in policy.state_dict().items()}
            stats = (norm.obs_rms.mean, norm.obs_rms.var, norm.clip_obs, norm.epsilon) if norm.norm_obs else \
                (0.0, 1.0 - norm.epsilon, np.inf, norm.epsilon)
            msg = (weights, stats, (norm.ret_rms.var, norm.clip_reward, norm.norm_reward), model.gamma, n,
                   model.sde_sample_freq)
            for c in self.conns:
                c.send(msg)
            parts = [c.recv() for c in self.conns]
            O, A, V, L, R, S, raw, returns, last_v, last_done, eps = zip(*parts)
            if norm.training:
                norm.obs_rms.update(np.concatenate(raw))
                norm.ret_rms.update(np.concatenate(returns))
            buf.reset()
            for name, arr in (("observations", O), ("actions", A), ("values", V), ("log_probs", L), ("rewards", R),
                              ("episode_starts", S)):
                getattr(buf, name)[:] = np.stack(arr, 1).reshape(getattr(buf, name).shape)
            buf.pos, buf.full = buf.buffer_size, True
            buf.compute_returns_and_advantage(torch.as_tensor(np.array(last_v, np.float32)), np.array(last_done))
            model.num_timesteps += n * len(self.conns)
            model._current_progress_remaining = 1.0 - model.num_timesteps / total
            model.train()
            episodes = (episodes + [e for part in eps for e in part])[-100:]
            for e in (e for part in eps for e in part):
                record = max(record, e.get("best_progress", -np.inf))
            log = model.logger
            if episodes:
                log.record("rollout/ep_rew_mean", safe_mean([e["r"] for e in episodes]))
                log.record("rollout/ep_len_mean", safe_mean([e["l"] for e in episodes]))
                log.record("rollout/best_progress_record", record)
                real = [e["best_progress"] for e in episodes if e.get("from_bottom", 1) and not e.get("rewinds", 0)]
                if real:
                    log.record("rollout/best_progress_mean", safe_mean(real))
            log.record("time/fps", int((model.num_timesteps - start_steps) / (time.time() - start_time)))
            log.record("time/total_timesteps", model.num_timesteps)
            log.dump(model.num_timesteps)
            if model.num_timesteps >= next_save:
                save(os.path.join(checkpoints, f"ppo_{model.num_timesteps}_steps.zip"))
                next_save += save_every

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
