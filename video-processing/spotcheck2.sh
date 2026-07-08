#!/usr/bin/env bash
# Spot-check head AND tail cuts for a few test files. Cuts are read from the
# saved region log re-scored with the new edge-tol; frames are grabbed just
# before/after each cut. Read-only; deletes local copy.
set -uo pipefail
VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"; T="$VP/sample_work"; OUT="$VP/output/spotcheck"; mkdir -p "$T" "$OUT"
LST=$(aws s3 ls s3://scua-video --recursive 2>/dev/null | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//')

shot(){ ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$2" -i "$1" -frames:v 1 -vf scale=240:180 "$3"; }
pair(){ ffmpeg -nostdin -y -hide_banner -loglevel error -i "$1" -i "$2" -filter_complex "[0:v][1:v]hstack" "$3"; }

for b in spc-mss1159-s02-i0087 RCC_20 RCC_82; do
  log="$VP/output/test/$b.probe.txt"
  ln=$("$PY" analyze_deadspace.py --from-log "$log" --edge-tol 5 2>&1 | grep "^\[mode")
  cs=$(echo "$ln" | sed -E 's/.*content ([0-9.]+)s ->.*/\1/')
  ce=$(echo "$ln" | sed -E 's/.*-> ([0-9.]+)s \|.*/\1/')
  echo "=== $b  head=$cs  tail=$ce ==="
  key=$(echo "$LST" | grep -F "/$b.mp4" | head -1)
  local="$T/$b.mp4"
  aws s3 cp "s3://scua-video/$key" "$local" --only-show-errors || { echo "  dl fail"; continue; }
  # HEAD: before (dead) | after (content)
  shot "$local" "$(awk "BEGIN{print ($cs>2)?$cs-2:0}")" "$T/a.png"
  shot "$local" "$(awk "BEGIN{print $cs+1.5}")" "$T/b.png"
  pair "$T/a.png" "$T/b.png" "$OUT/${b}_HEAD.png"
  # TAIL: before (content) | after (dead)
  shot "$local" "$(awk "BEGIN{print ($ce>1.5)?$ce-1.5:0}")" "$T/c.png"
  shot "$local" "$(awk "BEGIN{print $ce+2}")" "$T/d.png"
  pair "$T/c.png" "$T/d.png" "$OUT/${b}_TAIL.png"
  rm -f "$local" "$T"/a.png "$T"/b.png "$T"/c.png "$T"/d.png
  echo "  wrote ${b}_HEAD.png and ${b}_TAIL.png"
done
echo DONE
