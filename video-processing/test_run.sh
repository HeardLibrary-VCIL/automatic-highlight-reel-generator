#!/usr/bin/env bash
# End-to-end TEST of the full trimmer (black/freeze/silence + bars + snow) on a
# spread of distinct videos. Full downloads (snow needs the file's tail), runs
# report-only (no --apply), saves each region log, writes summary.tsv
# incrementally, deletes the local copy. Resumable: skips files already in the
# summary. Re-run after an SSO refresh to continue.
set -uo pipefail
VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"; BUCKET="s3://scua-video"
WORK="$VP/sample_work"; OUT="$VP/output/test"; mkdir -p "$WORK" "$OUT"
SUMMARY="$OUT/summary.tsv"; TARGET=18
[ -s "$SUMMARY" ] || printf "file\tdur_s\thead_s\ttail_s\tkept_pct\tstatus\tbars\tsnow\tnote\n" > "$SUMMARY"

aws s3 ls "$BUCKET" --recursive 2>/dev/null \
  | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//' \
  | grep -i '\.mp4$' | grep -iv '_trimmed\.mp4$' > "$OUT/all.txt"
awk -F/ '{b=$NF; if(!(b in s)){s[b]=1; print}}' "$OUT/all.txt" > "$OUT/uniq.txt"
total=$(wc -l < "$OUT/uniq.txt" | tr -d ' '); step=$(( total / TARGET )); [ "$step" -lt 1 ] && step=1
awk -v s="$step" 'NR % s == 1' "$OUT/uniq.txt" > "$OUT/selected.txt"
nsel=$(wc -l < "$OUT/selected.txt" | tr -d ' ')
echo "unique=$total step=$step selected=$nsel"

i=0
while IFS= read -r key <&3; do
  i=$((i+1)); base=$(basename "$key" .mp4)
  grep -q "^$base	" "$SUMMARY" && { echo "[$i/$nsel] $base (done, skip)"; continue; }
  echo "[$i/$nsel] $base"
  local="$WORK/$base.mp4"
  aws s3 cp "$BUCKET/$key" "$local" --only-show-errors \
    || { echo "  DL FAIL (creds expired?)"; printf "%s\tDL_FAIL\t\t\t\t\t\t\n" "$base" >> "$SUMMARY"; continue; }
  log="$OUT/${base}.probe.txt"
  "$PY" analyze_deadspace.py "$local" > "$log" 2>&1
  ln=$(grep "^\[mode" "$log" || true)
  if [ -z "$ln" ]; then printf "%s\tPROBE_FAIL\t\t\t\t\t\t\n" "$base" >> "$SUMMARY"; rm -f "$local"; continue; fi
  dur=$(grep -m1 "(duration" "$log" | sed -E 's/.*duration ([0-9.]+)s.*/\1/')
  hs=$(echo "$ln"|sed -E 's/.*content ([0-9.]+)s ->.*/\1/'); te=$(echo "$ln"|sed -E 's/.*-> ([0-9.]+)s \|.*/\1/')
  pct=$(echo "$ln"|sed -E 's/.*\(([0-9]+)%\).*/\1/'); st=$(echo "$ln"|sed -E 's/.*status: //')
  nb=$(grep -c " bars$" "$log"); nsno=$(grep -c " snow$" "$log")
  note=$(grep -m1 "  note:" "$log" | sed 's/.*note: //')
  echo "  $st kept ${pct}% head=$hs tail=$te bars=$nb snow=$nsno"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$base" "$dur" "$hs" "$te" "$pct" "$st" "$nb" "$nsno" "$note" >> "$SUMMARY"
  rm -f "$local"
done 3< "$OUT/selected.txt"
echo ""; echo "===== SUMMARY ====="; column -t -s $'\t' "$SUMMARY"
echo ""; echo "status counts:"; tail -n +2 "$SUMMARY" | cut -f6 | sort | uniq -c
echo DONE
