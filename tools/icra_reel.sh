#!/usr/bin/env bash
# Build the ICRA supplementary video: title, three architecture slides, the two
# head-to-head comparison segments, and the results table.
#
#   tools/icra_reel.sh                  # render everything, then assemble
#   tools/icra_reel.sh --assemble-only  # re-cut from segments already rendered
#
# `--assemble-only` re-cuts without paying for the two multi-minute comparison
# renders again -- use it when only a card, a slide or the running order changed.
#
# Output lands in results/icra_video/:
#   folbp_icra2027_reel.mp4         the cut, 1920x1080 30fps, 3:00
#   folbp_icra2027_reel_compact.mp4 the same cut at a submission-friendly size
#   folbp_icra2027_reel.srt         the burned-in captions, as a sidecar
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${COHERENT_PLANNER_PY:-$HOME/miniconda3/envs/coherent/bin/python}"

SIM_FOLBP=results/videos/house-folbp-20260826-171910
SIM_PEFA=results/videos/house-pefa-20260826-170555
OUT=results/icra_video
SEG=$OUT/segments
mkdir -p "$SEG"

# Identical on both sides of each comparison, which is the whole point: the two
# durations stay in their true ratio instead of being paced to taste.
SIM_SPEED=9.9      # wall-clock seconds per output second (both house runs)
HW_SPEED=4.6       # ditto, hardware footage

if [ "${1:-}" != "--assemble-only" ]; then
  echo "== cards =="
  for c in title sim hw; do
    "$PY" tools/compare_video.py card --spec "tools/icra/card_$c.json" \
          --out "$SEG/card_$c.mp4"
  done

  echo "== slides =="
  # The two architecture slides spotlight paper/fig/architecture.png -- the
  # figure the paper itself prints, not a redraw of it.
  for s in loop pipeline repair results; do
    "$PY" tools/explainer.py --spec "tools/icra/slide_$s.json" \
          --out "$SEG/slide_$s.mp4"
  done

  echo "== simulation, task 2 =="
  "$PY" tools/compare_video.py sim \
        --left "$SIM_FOLBP" --right "$SIM_PEFA" \
        --script tools/icra/sim_script.json \
        --out "$SEG/seg_sim.mp4" --speed "$SIM_SPEED" --tail 2.5

  echo "== real robots =="
  "$PY" tools/compare_video.py hw \
        --cues tools/icra/hw_cues.json \
        --out "$SEG/seg_hw.mp4" --speed "$HW_SPEED" --tail 4
fi

# Nine segments, 3:00. The architecture explainer is segments 2-4; everything
# after is footage the runs actually produced.
ORDER=(card_title slide_loop slide_pipeline slide_repair
       card_sim seg_sim card_hw seg_hw slide_results)

echo "== assemble =="
LIST="$SEG/concat.txt"
: > "$LIST"
for s in "${ORDER[@]}"; do
  [ -s "$SEG/$s.mp4" ] || { echo "missing segment: $SEG/$s.mp4" >&2; exit 1; }
  echo "file '$(basename "$s").mp4'" >> "$LIST"
done

# 3:00 is a ceiling, not a target: per-segment frame counts round up, and a file
# that probes at 180.03 s can fail a submission check that tests `<= 180`. TARGET
# pins the output under it; the guard below catches a running order that has
# genuinely outgrown the limit rather than just rounded past it.
TARGET=179.9
TOTAL=$(awk '{s+=$1} END {print s}' < <(
  for s in "${ORDER[@]}"; do
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$SEG/$s.mp4"
  done))
echo "[reel] segments total ${TOTAL}s -> trimming to ${TARGET}s"
awk -v t="$TOTAL" 'BEGIN { if (t > 181.0) {
  printf "[reel] WARNING: segments run %.2fs, more than a rounding overshoot.\n", t
  printf "[reel] The cut will be truncated. Shorten a segment instead.\n" } }'

# Three things beyond the picture, all of which a submission portal can trip on:
#   -t              keep the probed duration under the 3:00 ceiling
#   +faststart      move the moov atom to the front so it streams in a browser
#                   instead of needing the whole file before the first frame
#   silent AAC      a video-only mp4 is legal but some transcoders and slide
#                   embedders choke on a file with no audio stream at all
# Re-encoding rather than -c copy is unavoidable anyway: the stills and the busy
# segments come out of x264 at different levels and concat will not copy across
# that. level 4.0 is the widest-compatibility setting that still holds 1080p30.
ffmpeg -v error -stats -y -f concat -safe 0 -i "$LIST" \
       -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \
       -map 0:v:0 -map 1:a:0 -t "$TARGET" \
       -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -r 30 \
       -profile:v high -level 4.0 \
       -c:a aac -b:a 64k -movflags +faststart \
       "$OUT/folbp_icra2027_reel.mp4"

ffmpeg -v error -stats -y -i "$OUT/folbp_icra2027_reel.mp4" \
       -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p -r 30 \
       -profile:v high -level 4.0 \
       -c:a copy -movflags +faststart \
       "$OUT/folbp_icra2027_reel_compact.mp4"

# The tight one is two-pass to a bitrate rather than a quality target, because
# the constraint is the file size. Audio is mono at 24k: at 64k a silent track
# would spend 1.4 MB of a 10 MB budget saying nothing.
for pass in 1 2; do
  ffmpeg -v error -stats -y -i "$OUT/folbp_icra2027_reel.mp4" \
         -c:v libx264 -b:v 380k -pass $pass -preset veryslow \
         -pix_fmt yuv420p -profile:v high -level 4.0 \
         $( [ $pass = 1 ] && echo "-an -f mp4 /dev/null" \
                          || echo "-c:a aac -b:a 24k -ac 1 -movflags +faststart \
                                   $OUT/folbp_icra2027_reel_under10mb.mp4" )
done
rm -f ffmpeg2pass*.log*

# Sidecar captions. Same text and same timings as the burned-in band, offset by
# where each segment landed in the cut -- for editors that want them live, and
# for anyone watching with the picture too small to read the band.
"$PY" - "$SEG" "$OUT/folbp_icra2027_reel.srt" "${ORDER[@]}" <<'PY'
import json, subprocess, sys

seg_dir, out_path, order = sys.argv[1], sys.argv[2], sys.argv[3:]
SPECS = {"seg_sim":        ("tools/icra/sim_script.json", "captions"),
         "seg_hw":         ("tools/icra/hw_cues.json", "captions"),
         "slide_loop":     ("tools/icra/slide_loop.json", "captions"),
         "slide_pipeline": ("tools/icra/slide_pipeline.json", "captions"),
         "slide_repair":   ("tools/icra/slide_repair.json", "captions"),
         "slide_results":  ("tools/icra/slide_results.json", "captions")}


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def stamp(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return "%02d:%02d:%06.3f" % (h, m, s)


cues, offset = [], 0.0
for name in order:
    path = "%s/%s.mp4" % (seg_dir, name)
    dur = duration(path)
    if name in SPECS:
        spec_path, key = SPECS[name]
        caps = json.load(open(spec_path))[key]
        for i, (start, text) in enumerate(caps):
            end = caps[i + 1][0] if i + 1 < len(caps) else dur
            cues.append((offset + float(start), offset + min(float(end), dur), text))
    offset += dur

with open(out_path, "w") as fh:
    for i, (a, b, text) in enumerate(cues, 1):
        fh.write("%d\n%s --> %s\n%s\n\n"
                 % (i, stamp(a).replace(".", ","), stamp(b).replace(".", ","), text))
print("[reel] %d caption cues -> %s" % (len(cues), out_path))
PY

echo
echo "== done =="
for f in "$OUT"/folbp_icra2027_reel.mp4 "$OUT"/folbp_icra2027_reel_compact.mp4 \
         "$OUT"/folbp_icra2027_reel_under10mb.mp4; do
  printf "  %-46s %6.1f s  %6.1f MB\n" "$(basename "$f")" \
         "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")" \
         "$(echo "scale=1; $(stat -c %s "$f") / 1048576" | bc)"
done
