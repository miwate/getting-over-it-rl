#!/usr/bin/env bash
# Build the patched Box2D 2.3.1 plus goi_jax/reference/capi.cpp into build/box2d/libgoi_box2d.so, the CPU oracle
# loaded by goi_jax/reference/box2d.py. Fetches the sources first if needed (scripts/fetch_box2d.sh). A few seconds.
# Float flags: no FMA contraction and no fast-math, so every float32 op rounds once, as the game's MSVC x64 SSE
# build is assumed to do (overview R9). x86-64 gcc uses SSE scalar math by default (no x87 excess precision).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/third_party/box2d-2.3.1/Box2D"
OUT="$REPO_ROOT/build/box2d"
CXX="${CXX:-g++}"
FLAGS=(-O2 -ffp-contract=off -fno-fast-math -fPIC -std=c++11 -fno-strict-aliasing -DNDEBUG -I"$SRC")

[[ -f "$SRC/../.patched" ]] || "$REPO_ROOT/scripts/fetch_box2d.sh"
mkdir -p "$OUT/obj"
mapfile -t sources < <(cd "$SRC/Box2D" && find Collision Common Dynamics Rope -name '*.cpp' | sort)
objs=() pids=()
for s in "${sources[@]}"; do
    o="$OUT/obj/${s//\//_}.o"
    objs+=("$o")
    [[ "$o" -nt "$SRC/Box2D/$s" && "$o" -nt "$0" ]] && continue
    "$CXX" "${FLAGS[@]}" -w -c "$SRC/Box2D/$s" -o "$o" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done  # a bare `wait` would hide a failed compile
"$CXX" "${FLAGS[@]}" -Wall -Wextra -Wno-unused-parameter -shared -o "$OUT/libgoi_box2d.so" \
    "$REPO_ROOT/goi_jax/reference/capi.cpp" "${objs[@]}"
echo "DONE: $OUT/libgoi_box2d.so"
