#!/usr/bin/env bash
# Survey: for ~TARGET input videos, build a contact sheet (top row = 6 frames
# across the START, bottom row = 6 frames across the END) so we can eyeball and
# describe how leaders/trailers actually look. Read-only on S3; deletes the
# local copy after each. Resumable: skips files whose sheet already exists.
set -uo pipefail

VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"
BUCKET="s3://scua-video"
WORK="$VP/sample_work"
OUT="$VP/output/survey"
mkdir -p "$WORK" "$OUT"
TARGET=60

aws s3 ls "$BUCKET" --recursive 2>/dev/null \
  | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//' \
  | grep -i '\.mp4$' | grep -iv '_trimmed\.mp4$' > "$OUT/all_inputs.txt"
# Dedupe by basename first (the bucket has the same filename in many folders),
# so every Nth gives DISTINCT videos rather than colliding sheet names.
awk -F/ '{b=$NF; if(!(b in seen)){seen[b]=1; print}}' "$OUT/all_inputs.txt" > "$OUT/unique_inputs.txt"
total=$(wc -l < "$OUT/unique_inputs.txt" | tr -d ' ')
step=$(( total / TARGET )); [ "$step" -lt 1 ] && step=1
awk -v s="$step" 'NR % s == 1' "$OUT/unique_inputs.txt" > "$OUT/selected.txt"
nsel=$(wc -l < "$OUT/selected.txt" | tr -d ' ')
echo "total=$total step=$step selected=$nsel"

dur_of() { ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$1" 2>/dev/null \
           || ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=nokey=1:noprint_wrappers=1 "$1" 2>/dev/null; }
clamp() { awk -v t="$1" -v d="$2" 'BEGIN{ if(t<0.5)t=0.5; if(t>d-0.3)t=d-0.3; if(t<0)t=0; printf "%.2f", t }'; }

i=0
while IFS= read -r key <&3; do
  i=$((i+1)); base=$(basename "$key" .mp4)
  sheet="$OUT/${base}.png"
  if [ -s "$sheet" ]; then echo "[$i/$nsel] $base (have sheet, skip)"; continue; fi
  echo "[$i/$nsel] $base"
  local="$WORK/$base.mp4"
  aws s3 cp "$BUCKET/$key" "$local" --only-show-errors || { echo "  dl failed"; continue; }
  dur=$(dur_of "$local"); dur=${dur:-0}
  if awk -v d="$dur" 'BEGIN{exit !(d<5)}'; then echo "  bad/short duration ($dur)"; rm -f "$local"; continue; fi

  tmp="$WORK/frames_$base"; rm -rf "$tmp"; mkdir -p "$tmp"
  starts=(1 8 16 28 42 58); n=0
  for t in "${starts[@]}"; do n=$((n+1)); ts=$(clamp "$t" "$dur"); \
    ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$ts" -i "$local" -frames:v 1 -vf scale=200:150 "$tmp/$(printf '%02d' $n)_s.png"; done
  ends=(58 42 28 16 8 2)
  for t in "${ends[@]}"; do n=$((n+1)); ts=$(clamp "$(awk -v d="$dur" -v x="$t" 'BEGIN{printf "%.2f", d-x}')" "$dur"); \
    ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$ts" -i "$local" -frames:v 1 -vf scale=200:150 "$tmp/$(printf '%02d' $n)_e.png"; done

  ffmpeg -nostdin -y -hide_banner -loglevel error -framerate 1 -pattern_type glob -i "$tmp/*.png" \
    -frames:v 1 -vf "tile=6x2" "$sheet" 2>/dev/null \
    && echo "  wrote $(basename "$sheet") (dur=${dur}s)" || echo "  tile failed"

  rm -rf "$tmp"; rm -f "$local"
done 3< "$OUT/selected.txt"
echo "DONE"
