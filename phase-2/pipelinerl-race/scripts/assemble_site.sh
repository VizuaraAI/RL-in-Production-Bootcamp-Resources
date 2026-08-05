#!/usr/bin/env bash
# Assemble the deployable site from the verified pieces.
# Run AFTER scripts/publish_run.py has validated a run — this only copies.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf site/viz site/dashboard
cp -R viz site/viz
cp -R dashboard site/dashboard
echo "site/ assembled:"
find site -maxdepth 2 -name "index.html" | sed 's/^/  /'
[ -f site/viz/runs/conventional/events.jsonl ] && echo "  ✓ visualizer has run data" \
  || echo "  ! visualizer has NO run data — publish a run first"
[ -f site/dashboard/reference/summary.json ] && echo "  ✓ dashboard has reference summary" \
  || echo "  ! dashboard has NO reference — publish a run first"
