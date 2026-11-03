"""The Box2D step's sequential and per-fixture parts as Pallas kernels for GPUs (Triton backend; AMD ROCm tested).

    settings = dataclasses.replace(jb.Settings.game(), solver="pallas")
    run = jax.jit(jax.vmap(lambda md, s: jb.step(md, s, settings, dt, axis_name="w"), in_axes=(None, 0),
                           axis_name="w"))

Same inputs and outputs as the XLA step (Settings.solver == "pallas" is the switch; step.py and toi.py dispatch):
- kernel.py: the island solver (warm start, velocity iterations, position integration, position iterations with
  per-island early exit), one world per lane, worlds as the minor axis, all iterations in one launch;
- solve.py: b2World::Solve around it (set-up and wrap-up stay XLA);
- pairs.py: FindNewContacts' candidate scan and SynchronizeFixtures as kernels (the level's ~31k candidate pairs
  and ~1500 fixtures made these the largest per-world cost and memory in XLA);
- toi_search.py: b2World::SolveTOI's min-TOI search in one kernel (sweep alignment in list order, b2TimeOfImpact
  with b2Distance per queued contact, each lane working down its own queue, the minimum with the tie rule);
- toi_event.py: one TOI event in one kernel (advance to minAlpha, minContact->Update, the island walk with the
  narrow phase per lane, b2Island::SolveTOI's position and velocity passes, integration, report); the fixture sync
  and FindNewContacts after it use pairs.py's kernels. Before these, the XLA search (nested while loops over every
  contact lane) cost a fixed ~0.25 s per physics step whatever the batch (84% of a training rollout on the 3090);
- toi.py: b2Island::SolveTOI through the island kernel (the previous TOI path, kept; toi_event.py replaces it).
On a CPU backend the kernels run in Pallas interpret mode, bit-identical to the XLA step (tests/box2d/test_gpu.py);
on the GPU they matched the CPU XLA step bit for bit on player_rig, the TOI scenes and the level, except
velocities below FLT_MIN (XLA CPU flushes subnormals, the kernels keep them). Collision (narrow phase), contact
bookkeeping and islands of the main step stay in XLA. On the RTX 3090 (CUDA) GPU and CPU results agree to ~1e-6
in positions and ~1e-4 in velocities per step on the TOI scenes, identically with and without the TOI kernels
(the remaining differences come from the shared parts; sin/cos see kernel.py). Pass the Model as a jit argument,
not a closed-over constant: the level's Model as HLO constants costs minutes of XLA constant folding.
scripts/bench_gpu.py benchmarks; scripts/compile_toi.py times the TOI kernels' compile (~1-2 min each on CUDA).
"""

import os as _os

# Speed options (bit-identical results; off by default): GOI_GPU_SPEED is a comma list of
#   islands  the island search as a kernel (islands.py)
#   sort     worlds grouped by loop counts / position before the kernels (kernel.lane_order)
#   small    a smaller lane count for TOI loop iterations with few active worlds (toi_loop.SMALL_DIVISOR = 16)
#   idle     skip the start-of-step FindNewContacts when no world of the batch has a new fixture (step.py, sim.py)
#   place    FindNewContacts' placement of new pairs per slot (gather + selects) instead of batched scatters (pairs.py)
#   joints   the island solver computes a joint row for one type when the block's live lanes share it (kernel.py)
#   sync     the step's fixture sync reuses the island set-up's body rotations: one float-float sin/cos less (solve.py)
#   rays     the observation's ray casts as a kernel walking each world's own grid cell (goi_jax/game/rays_gpu.py)
#   proxy    the TOI search reads the contacts' convex proxies from one shared per-fixture table (toi_search.py)
#   graph    the training update's loops (rollout steps, physics sub-steps, GAE) stay inside the CUDA graph instead of
#            host-driven loops that read their predicate back each iteration (graph.py)
# or "all" = every option that measured faster on the RTX 3090 (runs/speed2/bench6_16k.log: training update at 16k
# envs 32.8k -> 46.8k env steps/s): sort, small, idle, place, joints, sync, proxy. "islands" and "rays" are bit-exact
# too but measured slower there (island-search kernel ~+1 s per update; rays kernel 0.69 s vs 0.42 s XLA), so "all"
# leaves them out; name them explicitly to use them. Read at import; the module attributes can be changed before
# tracing (scripts/bench_train.py --variants).
_SPEED = {t.strip() for t in _os.environ.get("GOI_GPU_SPEED", "").split(",") if t.strip()}
_ALL = ("sort", "small", "idle", "place", "joints", "sync", "proxy")


def speed_option(name):
    return name in _SPEED or ("all" in _SPEED and name in _ALL)

