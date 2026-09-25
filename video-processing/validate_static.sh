#!/usr/bin/env bash
# Validate the snow detector on real S3 files known (from the survey) to END in
# video static. Picks the smallest such files, full-downloads, scans the tail.
set -uo pipefail
VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"; T="$VP/sample_work"; mkdir -p "$T"

cands="RCC_118 RCC_128 RCC_148 RCC_162 RCC_160 RCC_87 RCC_178 RCC_184 RCC_23 \
spc-mss1159-s02-i0037 spc-mss1159-s02-i0081 spc-mss1159-s02-i0094 \
spc-mss1159-s02-i0135 spc-mss1159-s02-i0139 spc-mss1159-s02-i0146"

LST=$(aws s3 ls s3://scua-video --recursive 2>/dev/null)   # DATE TIME SIZE KEY
: > "$T/cand.tsv"
for b in $cands; do
  line=$(echo "$LST" | grep -F "/$b.mp4" | head -1) || true
  [ -z "$line" ] && continue
  size=$(echo "$line" | awk '{print $3}')
  key=$(echo "$line" | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//')
  printf "%s\t%s\t%s\n" "$size" "$b" "$key" >> "$T/cand.tsv"
done

echo "smallest snow-ending candidates:"; sort -n "$T/cand.tsv" | head -4 | awk -F'\t' '{printf "  %6.0f MB  %s\n",$1/1048576,$2}'
echo ""
sort -n "$T/cand.tsv" | head -4 | while IFS=$'\t' read -r size b key; do
  echo "=== $b ($(awk -v s="$size" 'BEGIN{printf "%.0fMB",s/1048576}')) ==="
  local="$T/$b.mp4"
  aws s3 cp "s3://scua-video/$key" "$local" --only-show-errors || { echo "  dl fail"; continue; }
  "$PY" detect_static.py "$local" --window 150 2>&1 | sed 's#.*: #  #'
  rm -f "$local"
done
echo DONE
