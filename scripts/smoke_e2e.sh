#!/usr/bin/env bash
# Headless end-to-end smoke test (RECREATE_SPEC §11 item 9 / §12 DoD).
# Runs the real CLI on a small area, asserts exit 0 and the 6 core artefacts.
set -euo pipefail
export MPLBACKEND=Agg

usage() { echo "usage: $0 [--city|--bbox] [extra cli args...]" >&2; exit 2; }
MODE="${1:-}"
[[ $# -gt 0 ]] && shift
case "$MODE" in
  --city) GEO=(--city "Otley, UK"); LABEL="otley"; BUDGET=200000 ;;
  --bbox) GEO=(--bbox 53.9065 53.9045 -1.6929 -1.6949); LABEL="bbox"; BUDGET=100000 ;;
  *) usage ;;
esac

OUT="$(mktemp -d -t bsp-smoke-XXXXXX)"
trap 'rm -rf "$OUT"' EXIT
PYBIN="${PYTHON:-python}"

echo "[$0] running: $PYBIN -m bike_rl.cli ${GEO[*]} --budget $BUDGET --timesteps 2048 --eval-episodes 2 --n-envs 2 --seed 0 --out-dir $OUT"
"$PYBIN" -m bike_rl.cli "${GEO[@]}" --budget "$BUDGET" \
  --timesteps 2048 --eval-episodes 2 --n-envs 2 --seed 0 --out-dir "$OUT"

# A run dir named <label>_<timestamp>/ sits under --out-dir
RUN_DIR="$(find "$OUT" -maxdepth 2 -mindepth 1 -type d | head -n1)"
echo "[$0] run_dir=$RUN_DIR"
missing=0
for f in final_model.zip run_summary.txt training_rewards.png \
         evaluation_metrics.png evaluation_rewards.png best_solution_map.png; do
  if [[ ! -f "$RUN_DIR/$f" ]]; then
    echo "[$0] MISSING artefact: $f" >&2; missing=1
  fi
done
grep -q '^best_reward:' "$RUN_DIR/run_summary.txt" || { echo "[$0] run_summary missing best_reward" >&2; missing=1; }
test "$missing" -eq 0
echo "[$0] OK: $LABEL smoke passed, 6 artefacts present"
