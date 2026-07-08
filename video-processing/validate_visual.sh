#!/usr/bin/env bash
# Visual accuracy validation: random sample of distinct videos, run the full
# trimmer, and emit head+tail boundary thumbnails so cuts can be eyeballed and
# scored. Full downloads (tail boundary needs the file's end). Resumable:
# skips files whose thumbnails already exist. Read-only; deletes local copy.
set -uo pipefail
VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"; BUCKET="s3://scua-video"
WORK="$VP/sample_work"; OUT="$VP/output/visual"; mkdir -p "$WORK" "$OUT"
MAN="$OUT/manifest.tsv"; TARGET=18
[ -s "$MAN" ] || printf "file\tdur_s\thead_s\ttail_s\tkept_pct\tstatus\n" > "$MAN"

# distinct basenames, then a reproducible-ish RANDOM sample
aws s3 ls "$BUCKET" --recursive 2>/dev/null \
  | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//' \
  | grep -i '\.mp4$' | grep -iv '_trimmed\.mp4$' > "$OUT/all.txt"
awk -F/ '{b=$NF; if(!(b in s)){s[b]=1; print}}' "$OUT/all.txt" \
  | awk 'BEGIN{srand(7)}{print rand()"\t"$0}' | sort -n | cut -f2- | head -"$TARGET" > "$OUT/selected.txt"
echo "selected $(wc -l < "$OUT/selected.txt" | tr -d ' ') random distinct videos"

shot(){ ffmpeg -nostdin -y -hide_banner -loglevel error -ss "$2" -i "$1" -frames:v 1 -vf scale=240:180 "$3" 2>/dev/null; }
pair(){ ffmpeg -nostdin -y -hide_banner -loglevel error -i "$1" -i "$2" -filter_complex "[0:v][1:v]hstack" "$3" 2>/dev/null; }

i=0; n=$(wc -l < "$OUT/selected.txt" | tr -d ' ')
while IFS= read -r key <&3; do
  i=$((i+1)); b=$(basename "$key" .mp4)
  [ -s "$OUT/${b}_HEAD.png" ] && { echo "[$i/$n] $b (have, skip)"; continue; }
  echo "[$i/$n] $b"
  local="$WORK/$b.mp4"
  aws s3 cp "$BUCKET/$key" "$local" --only-show-errors || { echo "  DL FAIL (creds?)"; continue; }
  ln=$("$PY" analyze_deadspace.py "$local" 2>/dev/null | grep "^\[mode" || true)
  [ -z "$ln" ] && { echo "  probe fail"; rm -f "$local"; continue; }
  dur=$("$PY" -c "import detect_static as d;print(round(d._video_duration('$local'),1))" 2>/dev/null)
  cs=$(echo "$ln"|sed -E 's/.*content ([0-9.]+)s ->.*/\1/'); ce=$(echo "$ln"|sed -E 's/.*-> ([0-9.]+)s \|.*/\1/')
  pct=$(echo "$ln"|sed -E 's/.*\(([0-9]+)%\).*/\1/'); st=$(echo "$ln"|sed -E 's/.*status: //')
  shot "$local" "$(awk "BEGIN{print ($cs>2)?$cs-2:0}")" "$WORK/a.png"; shot "$local" "$(awk "BEGIN{print $cs+1.5}")" "$WORK/b.png"
  pair "$WORK/a.png" "$WORK/b.png" "$OUT/${b}_HEAD.png"
  shot "$local" "$(awk "BEGIN{print ($ce>1.5)?$ce-1.5:0}")" "$WORK/c.png"; shot "$local" "$(awk "BEGIN{d=$dur; t=$ce+2; print (t<d)?t:d-0.3}")" "$WORK/e.png"
  pair "$WORK/c.png" "$WORK/e.png" "$OUT/${b}_TAIL.png"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$b" "$dur" "$cs" "$ce" "$pct" "$st" >> "$MAN"
  echo "  $st kept ${pct}% head=$cs tail=$ce"
  rm -f "$local" "$WORK"/a.png "$WORK"/b.png "$WORK"/c.png "$WORK"/e.png
done 3< "$OUT/selected.txt"
echo ""; column -t -s $'\t' "$MAN"; echo DONE
