#!/usr/bin/env bash
# Spot-check: for a few OK files, re-download and grab frames just before/after
# the proposed head and tail cuts, so we can eyeball that the black-anchor lands
# at a real program boundary. Reads cut times from output/sample2/summary.tsv.
set -uo pipefail

VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
BUCKET="s3://scua-video"
WORK="$VP/sample_work"
OUT="$VP/output/spotcheck"
SUM="$VP/output/sample2/summary.tsv"
mkdir -p "$WORK" "$OUT"

# files to inspect (all status OK, non-trivial head+tail trims, both naming types)
bases=( spc-mss1159-s02-i0128 spc-mss1159-s02-i0158 RCC_134 RCC_85 )

# build base -> full S3 key map from the listing
aws s3 ls "$BUCKET" --recursive 2>/dev/null \
  | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//' > "$OUT/keys.txt"

for base in "${bases[@]}"; do
  echo "=================== $base ==================="
  row=$(awk -F'\t' -v b="$base" '$1==b{print; exit}' "$SUM")
  dur=$(echo "$row"  | cut -f2); hs=$(echo "$row" | cut -f3); te=$(echo "$row" | cut -f4)
  pct=$(echo "$row"  | cut -f5)
  echo "proposed: head=${hs}s tail=${te}s (dur ${dur}s, kept ${pct}%)"

  key=$(grep -F "/$base.mp4" "$OUT/keys.txt" | head -1)
  if [ -z "$key" ]; then echo "  key not found"; continue; fi
  local="$WORK/$base.mp4"
  aws s3 cp "$BUCKET/$key" "$local" --only-show-errors || { echo "  dl failed"; continue; }

  # head: ~1.5s before cut (should be dead) | ~1s after cut (should be content)
  hb=$(awk "BEGIN{print ($hs>1.5)?$hs-1.5:0}")
  ha=$(awk "BEGIN{print $hs+1.0}")
  ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$hb" -i "$local" -frames:v 1 -vf scale=320:240 "$WORK/hb.png"
  ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$ha" -i "$local" -frames:v 1 -vf scale=320:240 "$WORK/ha.png"
  ffmpeg -nostdin -y -hide_banner -loglevel error -i "$WORK/hb.png" -i "$WORK/ha.png" \
    -filter_complex "[0:v][1:v]hstack" "$OUT/${base}_HEAD.png"

  # tail: ~1s before cut (should be content) | ~1.5s after cut (should be dead)
  tb=$(awk "BEGIN{print ($te>1.0)?$te-1.0:0}")
  ta=$(awk "BEGIN{print $te+1.5}")
  ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$tb" -i "$local" -frames:v 1 -vf scale=320:240 "$WORK/tb.png"
  ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$ta" -i "$local" -frames:v 1 -vf scale=320:240 "$WORK/ta.png"
  ffmpeg -nostdin -y -hide_banner -loglevel error -i "$WORK/tb.png" -i "$WORK/ta.png" \
    -filter_complex "[0:v][1:v]hstack" "$OUT/${base}_TAIL.png"

  rm -f "$local" "$WORK"/hb.png "$WORK"/ha.png "$WORK"/tb.png "$WORK"/ta.png
  echo "  wrote ${base}_HEAD.png and ${base}_TAIL.png"
done
echo "DONE"
