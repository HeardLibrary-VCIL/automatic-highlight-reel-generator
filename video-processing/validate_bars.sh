#!/usr/bin/env bash
# Validate the bars detector on real S3 files using cheap partial (~35MB)
# downloads. POS = should detect bars; TRAP = static/program, must NOT.
set -uo pipefail
VP=/Users/lingxinchen/automatic-highlight-reel-generator/video-processing
cd "$VP" || exit 1
PY="$VP/.venv/bin/python"; T="$VP/sample_work/bars_test"; mkdir -p "$T"
aws s3 ls s3://scua-video --recursive 2>/dev/null | sed -E 's/^[0-9-]+ +[0-9:]+ +[0-9]+ +//' > "$T/keys.txt"

# base  label
items=(
"RCC_87:POS(full-minute bars)"
"RCC_174:POS(bars→title→black)"
"RCC_181:POS(bars→Coast#67)"
"RCC_178:POS(bars→Coast#56)"
"RCC_98:POS(bars→CommunityAccess)"
"spc-mss1159-s02-i0094:POS(bars→LIFESTYLES slate)"
"spc-mss1159-s02-i0064:POS(black→bars+9)"
"RCC_73:TRAP(dark hallway program)"
"RCC_164:TRAP(host straight in)"
"RCC_160:TRAP(black→crowd program)"
"spc-mss1159-s02-i0046:TRAP(interview, no bars)"
)
printf "%-26s %-26s %s\n" "FILE" "EXPECTED" "DETECTED"
for it in "${items[@]}"; do
  base="${it%%:*}"; label="${it#*:}"
  key=$(grep -F "/$base.mp4" "$T/keys.txt" | head -1)
  [ -z "$key" ] && { printf "%-26s %-26s %s\n" "$base" "$label" "KEY-NOT-FOUND"; continue; }
  part="$T/part.mp4"
  aws s3api get-object --bucket scua-video --key "$key" --range "bytes=0-35000000" "$part" >/dev/null 2>&1 \
    || { printf "%-26s %-26s %s\n" "$base" "$label" "DL-FAIL"; continue; }
  out=$("$PY" detect_bars.py "$part" --window 90 2>&1 | tail -1)
  res=$(echo "$out" | sed -E 's#.*sampled frames ##')
  printf "%-26s %-26s %s\n" "$base" "$label" "$res"
  rm -f "$part"
done
echo DONE
