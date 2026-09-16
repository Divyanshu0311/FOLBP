#!/usr/bin/env python3
"""Build a step-progress overlay for hand-recorded hardware footage.

    hw_overlay.py --run_dir results/realrobot_tb/hw_runs/<run>          # inspect
    hw_overlay.py --run_dir <run> --anchor 0=4.2 --anchor 4=71.0 --end 118
    hw_overlay.py --run_dir <run> --anchors anchors.json --burn demo.mp4

`compose_video.py` joins sim frames to planner events on `time.time()`. Neither
half of that exists for the real-robot runs: the footage came off a camera that
never saw the planner, and the transcripts carry per-step DURATIONS but no
absolute clock. So the video itself is the clock here -- you scrub to where an
action visibly starts, and everything between anchors is filled in using the
durations the run actually logged.

Anchor as few or as many steps as you like. One anchor plus the logged durations
already places every cue; the reason to add more is that the gap between two
steps is the planner thinking, and that time was never recorded, so a single
anchor drifts. Anchoring the first step of each agent is usually enough.

Output is an .ass checklist that accumulates -- done steps tick green, the
running one is highlighted, the rest stay dim -- plus an .srt fallback for
editors that will not take ASS.
"""
import argparse
import json
import os
import re
import subprocess
import sys

# ------------------------------------------------------------------ appearance
# Same palette as compose_video.py so the two demo styles stay siblings.
# ASS wants &HBBGGRR&, not RGB.
A_FG = "&HD9D1C9&"
A_DONE = "&H50B93F&"
A_NOW = "&HFFA658&"
A_PEND = "&H81766E&"

MARK_DONE = "✓"
MARK_NOW = "▶"
MARK_PEND = "·"


# ------------------------------------------------------------------ parsing

RE_TIMING = re.compile(r"^\s*([\d.]+)s\s+(\w+)\s+(.+?)\s*->\s*(.+?)\s*$")
RE_FOLBP_EXEC = re.compile(r"\[FOLBP\]\[EXEC\] step\[(\d+)\]\s+(.+?)\s*->\s*(.+?)\s*$")
RE_FOLBP_TOOK = re.compile(r"\[FOLBP\]\[TB\] step took ([\d.]+)s \((\w+)\)")
RE_FOLBP_KIND = re.compile(r"\[FOLBP\]\[TB\] (send|manual) ")
RE_PEFA_LOG = re.compile(r"^\s*Step\s+(\d+):\s+(.+?)\s*->\s*(.+?)\s*$")


def parse_steps(text):
    """-> [{'agent','action','kind','dur'}], in execution order.

    Three sources, in descending order of trust. The Execution Timing block is
    authoritative when present (it is what the dispatcher actually measured);
    FOLBP's per-step EXEC/TB lines carry the same facts scattered; the
    Step-by-Step Action Log is a last resort and has no durations at all.
    """
    lines = text.splitlines()

    timing = _from_timing_block(lines)
    if timing:
        return timing

    folbp = _from_folbp_exec(lines)
    if folbp:
        return folbp

    return _from_action_log(lines)


def _from_timing_block(lines):
    try:
        start = next(i for i, l in enumerate(lines) if "=== Execution Timing ===" in l)
    except StopIteration:
        return []
    out = []
    for line in lines[start + 1:]:
        if line.startswith("  device ") or not line.strip():
            break
        m = RE_TIMING.match(line)
        if not m:
            break
        dur, kind, agent, action = m.groups()
        out.append({"agent": agent, "action": action, "kind": kind, "dur": float(dur)})
    return out


def _from_folbp_exec(lines):
    out = []
    for line in lines:
        m = RE_FOLBP_EXEC.search(line)
        if m:
            _, agent, action = m.groups()
            out.append({"agent": agent, "action": action, "kind": None, "dur": None})
            continue
        if not out:
            continue
        m = RE_FOLBP_KIND.search(line)
        if m and out[-1]["kind"] is None:
            out[-1]["kind"] = "manual" if m.group(1) == "manual" else "http"
        m = RE_FOLBP_TOOK.search(line)
        if m:
            out[-1]["dur"] = float(m.group(1))
            out[-1]["kind"] = m.group(2)
    return out


def _from_action_log(lines):
    try:
        start = next(i for i, l in enumerate(lines)
                     if "=== Step-by-Step Action Log ===" in l)
    except StopIteration:
        return []
    out = []
    for line in lines[start + 1:]:
        m = RE_PEFA_LOG.match(line)
        if not m:
            break
        _, agent, action = m.groups()
        out.append({"agent": agent, "action": action, "kind": None, "dur": None})
    return out


def tidy(s):
    """`<robot dog>(400)` and `robot dog(400)` both -> `robot dog`."""
    s = re.sub(r"[<>]", "", s)
    return re.sub(r"\(\d+\)", "", s).strip()


def label(step):
    verb = re.sub(r"[\[\]]", "", re.match(r"\[?([a-z_]+)\]?", step["action"]).group(1))
    target = step["action"].split("]", 1)[-1].strip()
    return "%s: %s %s" % (tidy(step["agent"]), verb.replace("_", " "), tidy(target))


# ------------------------------------------------------------------ placement

def place(steps, anchors, end=None, gap=0.0):
    """Anchors (step index -> video seconds) + logged durations -> cue starts.

    An anchor is ground truth and is never moved. Between two anchors the
    recorded durations set the proportions, so a step the arm spent 26s on gets
    26s worth of the interval; unmeasured runs fall back to equal shares. Before
    the first anchor and after the last, durations are used directly with `gap`
    standing in for the planner's unrecorded thinking time.
    """
    n = len(steps)
    if not anchors:
        raise ValueError("need at least one --anchor")

    durs = [s["dur"] if s["dur"] else 0.0 for s in steps]
    if all(d == 0.0 for d in durs):
        durs = [1.0] * n

    starts = [None] * n
    for i, t in anchors.items():
        if not 0 <= i < n:
            raise ValueError("anchor %d is outside the run's %d steps" % (i, n))
        starts[i] = float(t)

    known = sorted(anchors)

    # Between consecutive anchors: split the interval by logged duration.
    for a, b in zip(known, known[1:]):
        span = starts[b] - starts[a]
        if span <= 0:
            raise ValueError("anchor %d (%.2fs) is not after anchor %d (%.2fs)"
                             % (b, starts[b], a, starts[a]))
        weight = sum(durs[a:b]) or float(b - a)
        t = starts[a]
        for i in range(a, b):
            starts[i] = t
            t += span * (durs[i] / weight)

    # Outside the anchored range: durations plus a nominal deliberation gap.
    for i in range(known[0] - 1, -1, -1):
        starts[i] = starts[i + 1] - (durs[i] + gap)
    for i in range(known[-1] + 1, n):
        starts[i] = starts[i - 1] + (durs[i - 1] + gap)

    if starts[0] < 0:
        raise ValueError("anchors put step 0 at %.2fs, before the video starts"
                         % starts[0])

    last_end = end if end is not None else starts[-1] + max(durs[-1], 2.0) + gap
    if last_end <= starts[-1]:
        raise ValueError("--end %.2fs is not after the last step at %.2fs"
                         % (last_end, starts[-1]))
    return starts, float(last_end)


# ------------------------------------------------------------------ rendering

def ts_ass(t):
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%05.2f" % (int(h), int(m), s)


def ts_srt(t):
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return ("%02d:%02d:%06.3f" % (int(h), int(m), s)).replace(".", ",")


def checklist(steps, cur):
    """The whole list every time, with everything up to `cur` already ticked."""
    out = []
    for i, st in enumerate(steps):
        if i < cur:
            mark, colour = MARK_DONE, A_DONE
        elif i == cur:
            mark, colour = MARK_NOW, A_NOW
        else:
            mark, colour = MARK_PEND, A_PEND
        out.append(r"{\c%s}%s %d. %s" % (colour, mark, i + 1, label(st)))
    return r"\N".join(out)


def build_ass(steps, starts, last_end, title, font_size, margin):
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
        "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding",
        "Style: List,DejaVu Sans Mono,%d,%s,&H00000000,&HA0000000,0,0,0,0,"
        "100,100,0,0,3,0,0,1,%d,40,%d,1" % (font_size, A_FG, margin, margin),
        "Style: Title,DejaVu Sans Mono,%d,%s,&H00000000,&HA0000000,1,0,0,0,"
        "100,100,0,0,3,0,0,7,%d,40,40,1" % (int(font_size * 0.9), A_FG, margin),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    ev = []
    if title:
        ev.append("Dialogue: 0,%s,%s,Title,,0,0,0,,%s"
                  % (ts_ass(starts[0]), ts_ass(last_end), title))
    bounds = list(starts[1:]) + [last_end]
    for i, (a, b) in enumerate(zip(starts, bounds)):
        ev.append("Dialogue: 0,%s,%s,List,,0,0,0,,%s"
                  % (ts_ass(a), ts_ass(b), checklist(steps, i)))
    return "\n".join(head + ev) + "\n"


def build_srt(steps, starts, last_end):
    bounds = list(starts[1:]) + [last_end]
    out = []
    for i, (a, b) in enumerate(zip(starts, bounds)):
        body = "\n".join(
            "%s %d. %s" % (MARK_DONE if j < i else MARK_NOW if j == i else MARK_PEND,
                           j + 1, label(st))
            for j, st in enumerate(steps))
        out.append("%d\n%s --> %s\n%s\n" % (i + 1, ts_srt(a), ts_srt(b), body))
    return "\n".join(out)


# ------------------------------------------------------------------ cli

def parse_anchor_args(items, path):
    anchors = {}
    if path:
        with open(path) as fh:
            for k, v in json.load(fh).items():
                anchors[int(k)] = float(v)
    for it in items or []:
        if "=" not in it:
            raise SystemExit("--anchor wants STEP=SECONDS, got %r" % it)
        k, v = it.split("=", 1)
        anchors[int(k)] = parse_time(v)
    return anchors


def parse_time(v):
    """`71`, `71.5` or `1:11.5`."""
    v = v.strip()
    if ":" not in v:
        return float(v)
    m, s = v.rsplit(":", 1)
    return float(m) * 60.0 + float(s)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", help="an hw_runs/<run> directory (reads transcript.txt)")
    src.add_argument("--transcript", help="a transcript file directly")
    ap.add_argument("--anchor", action="append", metavar="STEP=TIME",
                    help="step index (0-based) = video time, e.g. 0=4.2 or 4=1:11.5. "
                         "Repeatable.")
    ap.add_argument("--anchors", help="JSON file of the same, {\"0\": 4.2, ...}")
    ap.add_argument("--end", help="video time the last step finishes")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="seconds of planner deliberation to assume between steps "
                         "outside the anchored range (default 0)")
    ap.add_argument("--title", default=None, help="header line; default from --run_dir")
    ap.add_argument("--out", default=None, help="output .ass (default <run_dir>/overlay.ass)")
    ap.add_argument("--font_size", type=int, default=34)
    ap.add_argument("--margin", type=int, default=60)
    ap.add_argument("--burn", metavar="VIDEO",
                    help="also run ffmpeg to burn the overlay into this file")
    ap.add_argument("--burn_out", default=None, help="output for --burn")
    args = ap.parse_args()

    path = args.transcript or os.path.join(args.run_dir, "transcript.txt")
    with open(path) as fh:
        text = fh.read()

    steps = parse_steps(text)
    if not steps:
        raise SystemExit("no steps found in %s" % path)

    measured = sum(1 for s in steps if s["dur"])
    print("%d steps from %s (%d with logged durations)"
          % (len(steps), path, measured))
    for i, s in enumerate(steps):
        print("  %d  %-9s %-7s %s"
              % (i, ("%.2fs" % s["dur"]) if s["dur"] else "--",
                 s["kind"] or "?", label(s)))

    anchors = parse_anchor_args(args.anchor, args.anchors)
    if not anchors:
        print("\nNo anchors given, so nothing was written.")
        print("Scrub the footage to where a step's motion starts and pass it in:")
        print("    --anchor 0=<seconds>            one anchor is the minimum")
        print("    --anchor 0=<s> --anchor %d=<s>   one per agent is usually enough"
              % next((i for i, s in enumerate(steps) if s["kind"] == "manual"),
                     len(steps) - 1))
        print("    --end <seconds>                 when the last step finishes")
        return

    starts, last_end = place(steps, anchors, 
                             parse_time(args.end) if args.end else None, args.gap)

    title = args.title
    if title is None and args.run_dir:
        title = os.path.basename(os.path.normpath(args.run_dir))

    out = args.out or (os.path.join(args.run_dir, "overlay.ass")
                       if args.run_dir else "overlay.ass")
    with open(out, "w") as fh:
        fh.write(build_ass(steps, starts, last_end, title, args.font_size, args.margin))
    srt = os.path.splitext(out)[0] + ".srt"
    with open(srt, "w") as fh:
        fh.write(build_srt(steps, starts, last_end))

    print("\ncue sheet (anchored steps marked *):")
    bounds = list(starts[1:]) + [last_end]
    for i, (a, b) in enumerate(zip(starts, bounds)):
        print("  %d  %7.2f -> %7.2f  %s%s"
              % (i, a, b, "* " if i in anchors else "  ", label(steps[i])))
    print("\nwrote %s\nwrote %s" % (out, srt))

    if not args.burn:
        print("\nburn it in with:")
        print("    ffmpeg -i <video> -vf \"ass=%s\" -c:a copy <out>.mp4" % out)
        return

    burn_out = args.burn_out or (os.path.splitext(args.burn)[0] + "-annotated.mp4")
    cmd = ["ffmpeg", "-y", "-i", args.burn, "-vf", "ass=%s" % out,
           "-c:a", "copy", burn_out]
    print("\n$ %s" % " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc:
        raise SystemExit("ffmpeg failed (%d)" % rc)
    print("wrote %s" % burn_out)


if __name__ == "__main__":
    main()
