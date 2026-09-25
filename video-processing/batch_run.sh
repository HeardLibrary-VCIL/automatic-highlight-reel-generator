#!/usr/bin/env bash
# Bigger validation run: select ~TARGET input videos spread across the bucket,
# download -> probe (report only, black-anchored) -> record status -> delete.
# Writes summary.tsv incrementally so partial results survive a creds expiry.
set -uo pipefail

VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"
BUCKET="s3://scua-video"
WORK="$VP/sample_work"
OUT="$VP/output/sample2"
mkdir -p "$WORK" "$OUT"
SUMMARY="$OUT/summary.tsv"
printf "file\tdur_s\thead_s\ttail_s\tkept_pct\tstatus\tnote\n" > "$SUMMARY"
TARGET=30

# List input keys (strip "date time size " prefix; keep keys with spaces), exclude _trimmed outputs.
aws s3 ls "$BUCKET" --recursive 2>/dev/null \
  | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//' \
  | grep -i '\.mp4$' | grep -iv '_trimmed\.mp4$' > "$OUT/all_inputs.txt"
total=$(wc -l < "$OUT/all_inputs.txt" | tr -d ' ')
step=$(( total / TARGET )); [ "$step" -lt 1 ] && step=1
awk -v s="$step" 'NR % s == 1' "$OUT/all_inputs.txt" > "$OUT/selected.txt"
nsel=$(wc -l < "$OUT/selected.txt" | tr -d ' ')
echo "total inputs=$total  step=$step  selected=$nsel"
echo ""

i=0
while IFS= read -r key <&3; do
  i=$((i+1))
  base=$(basename "$key" .mp4)
  echo "[$i/$nsel] $base"
  local="$WORK/$base.mp4"
  if ! aws s3 cp "$BUCKET/$key" "$local" --only-show-errors; then
    echo "  DOWNLOAD FAILED (creds expired?)"
    printf "%s\tDOWNLOAD_FAILED\t\t\t\t\n" "$base" >> "$SUMMARY"; continue
  fi
  log="$OUT/${base}.probe.txt"
  "$PY" analyze_deadspace.py "$local" > "$log" 2>&1
  ln=$(grep "^\[mode" "$log" || true)
  if [ -z "$ln" ]; then
    echo "  PROBE FAILED"
    printf "%s\tPROBE_FAILED\t\t\t\t\n" "$base" >> "$SUMMARY"; rm -f "$local"; continue
  fi
  dur=$(grep -m1 "(duration"  "$log" | sed -E 's/.*duration ([0-9.]+)s.*/\1/')
  hs=$(echo "$ln"   | sed -E 's/.*content ([0-9.]+)s ->.*/\1/')
  te=$(echo "$ln"   | sed -E 's/.*-> ([0-9.]+)s \|.*/\1/')
  pct=$(echo "$ln"  | sed -E 's/.*\(([0-9]+)%\).*/\1/')
  st=$(echo "$ln"   | sed -E 's/.*status: //')
  note=$(grep -m1 "  note:" "$log" | sed 's/.*note: //')
  echo "  $st  kept ${pct}%  ($hs -> $te)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$base" "$dur" "$hs" "$te" "$pct" "$st" "$note" >> "$SUMMARY"
  rm -f "$local"
done 3< "$OUT/selected.txt"

echo ""
echo "===== SUMMARY ====="
column -t -s $'\t' "$SUMMARY"
echo ""
echo "status counts:"; tail -n +2 "$SUMMARY" | cut -f6 | sort | uniq -c
