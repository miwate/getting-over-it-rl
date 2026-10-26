#!/usr/bin/env bash
# Fetch Box2D v2.3.1 (the release Unity's Physics2D fork is believed to come from) into third_party/box2d-2.3.1 and
# apply scripts/patches/*.patch, which turn the tuning constants Unity exposes into runtime globals and add Unity's
# TOI rule. The result is the CPU oracle the JAX engine is checked against; build it with scripts/build_box2d.sh.
#   scripts/fetch_box2d.sh            # no-op if already fetched
#   scripts/fetch_box2d.sh --force    # re-extract and re-patch (local edits under third_party are lost)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://codeload.github.com/erincatto/box2d/tar.gz/refs/tags/v2.3.1"
# codeload archives are generated on the fly; if GitHub ever re-packs this one, check the tree before updating the hash.
SHA256="58ffc8475a8650aadc351345aef696937747b40501ab78d72c197c5ff5b3035c"
DEST="$REPO_ROOT/third_party/box2d-2.3.1"

if [[ -f "$DEST/.patched" && "${1:-}" != "--force" ]]; then echo "already fetched: $DEST"; exit 0; fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/box2d.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
curl -sSLf -o "$tmp/box2d.tar.gz" "$URL"
echo "$SHA256  $tmp/box2d.tar.gz" | sha256sum -c --quiet -

rm -rf "$DEST"
mkdir -p "$DEST"
tar xzf "$tmp/box2d.tar.gz" -C "$DEST" --strip-components=1
# Patches are relative to the Box2D/ folder (the one holding Box2D/Box2D.h). Sources use CRLF; patches keep it.
for p in "$REPO_ROOT"/scripts/patches/box2d-2.3.1-*.patch; do
    patch -s -p1 -d "$DEST/Box2D" < "$p"
    echo "applied $(basename "$p")"
done
touch "$DEST/.patched"
echo "DONE: $DEST"
