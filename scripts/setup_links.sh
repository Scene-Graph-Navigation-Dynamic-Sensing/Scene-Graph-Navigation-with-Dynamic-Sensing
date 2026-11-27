#!/usr/bin/env bash
# Link a fresh clone of this repo to the shared data on the lab server.
#   bash scripts/setup_links.sh            # uses /home/ravindu/SceneGraphNav
#   bash scripts/setup_links.sh /path/to/SceneGraphNav
# Creates data/ models/ external/ results/ outputs/ in the repo as symlinks to the shared folders.
# Pass --own-results to keep results/ and outputs/ private to this clone instead.
set -euo pipefail
SHARED=/home/ravindu/SceneGraphNav; OWN=0
for a in "$@"; do case $a in --own-results) OWN=1;; *) SHARED=$a;; esac; done
cd "$(dirname "$0")/.."
link() { local name=$1 target=$2
  if [ -L "$name" ]; then echo "ok    $name -> $(readlink "$name")"; return; fi
  if [ -e "$name" ]; then echo "SKIP  $name exists and is not a link"; return; fi
  ln -s "$target" "$name"; echo "link  $name -> $target"; }
link data     "$SHARED/datasets"
link models   "$SHARED/models"
link external "$SHARED/external"
if [ $OWN = 1 ]; then mkdir -p results outputs; echo "own   results/ outputs/"
else link results "$SHARED/results"; link outputs "$SHARED/outputs"; fi
link mobileclip2_b.ts models/checkpoints/mobileclip2_b.ts
