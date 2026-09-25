#!/usr/bin/env bash
# Validation-sample driver: for each S3 key, download -> probe (report only) ->
# capture proposed cut -> make a before/after-head-cut thumbnail -> delete local copy.
set -uo pipefail

VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"
BUCKET="s3://scua-video"
WORK="$VP/sample_work"
OUT="$VP/output/sample"
mkdir -p "$WORK" "$OUT"
SUMMARY="$OUT/summary.tsv"
printf "file\tduration_s\thead_cut_s\ttail_cut_s\tkept_s\n" > "$SUMMARY"

keys=(
"RainbowCommunityCenter/41/Web Copy/RCC_41.mp4"
"RainbowCommunityCenter/4/97/Web Copy/RCC_97.mp4"
"RainbowCommunityCenter/4/Web Copy/RCC_4.mp4"
"RainbowCommunityCenter/6/Web Copy/RCC_6.mp4"
"RainbowCommunityCenter/43/Web Copy/spc-mss1159-s02-i0043.mp4"
"RainbowCommunityCenter/5/Web Copy/spc-mss1159-s02-i0005.mp4"
"RainbowCommunityCenter/46/Web Copy/spc-mss1159-s02-i0046.mp4"
"RainbowCommunityCenter/61/Web Copy/spc-mss1159-s02-i0061.mp4"
)

for key in "${keys[@]}"; do
  base=$(basename "$key" .mp4)
  echo "=================== $base ==================="
  local="$WORK/$base.mp4"
  echo "downloading: $key"
  if ! aws s3 cp "$BUCKET/$key" "$local" --only-show-errors; then
    echo "DOWNLOAD FAILED"; printf "%s\tDOWNLOAD_FAILED\t\t\t\n" "$base" >> "$SUMMARY"; continue
  fi

  log="$OUT/${base}.probe.txt"
  "$PY" analyze_deadspace.py "$local" > "$log" 2>&1
  line=$(grep "Proposed content window" "$log" || true)
  echo "$line"
  if [ -z "$line" ]; then
    echo "PROBE FAILED (see $log)"; printf "%s\tPROBE_FAILED\t\t\t\n" "$base" >> "$SUMMARY"; rm -f "$local"; continue
  fi
  dur=$(grep -m1 "(duration" "$log" | sed -E 's/.*duration ([0-9.]+)s.*/\1/')
  hs=$(echo "$line"  | sed -E 's/.*window: ([0-9.]+)s ->.*/\1/')
  te=$(echo "$line"  | sed -E 's/.*-> ([0-9.]+)s .*/\1/')
  kept=$(awk "BEGIN{printf \"%.1f\", $te-$hs}")
  printf "%s\t%s\t%s\t%s\t%s\n" "$base" "$dur" "$hs" "$te" "$kept" >> "$SUMMARY"

  # before head cut (should be dead) | after head cut (should be content)
  before_t=$(awk "BEGIN{print ($hs>2)?$hs-2:0}")
  after_t=$(awk "BEGIN{print $hs+0.5}")
  ffmpeg -y -hide_banner -loglevel error -ss "$before_t" -i "$local" -frames:v 1 -vf "scale=320:240" "$WORK/_b.png"
  ffmpeg -y -hide_banner -loglevel error -ss "$after_t"  -i "$local" -frames:v 1 -vf "scale=320:240" "$WORK/_a.png"
  ffmpeg -y -hide_banner -loglevel error -i "$WORK/_b.png" -i "$WORK/_a.png" \
    -filter_complex "[0:v][1:v]hstack" "$OUT/${base}_headcut.png" 2>/dev/null

  rm -f "$local" "$WORK/_b.png" "$WORK/_a.png"
  echo "done; removed local copy"
done

echo ""
echo "===== SUMMARY (left=before cut/should be dead, right=after cut/should be content) ====="
column -t -s $'\t' "$SUMMARY"
