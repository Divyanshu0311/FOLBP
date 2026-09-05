#!/usr/bin/env python3
"""Compose a demo video: sim footage on top, live planner transcript underneath.

    compose_video.py --run_dir results/videos/<run> --framework folbp

Inputs, all produced during the run and joined here on wall clock:

    sim.mp4        the viewer camera, written by Benchmark/recorder.py  (py3.7)
    frames.jsonl   {"i": <frame index>, "t": <time.time()>} per frame
    events.jsonl   {"t": ..., "stage": ..., "title": ..., "body": ...} per stage,
                   written by bench/eventlog.py                        (py3.10)

The two processes never talk about recording; `time.time()` is the entire
synchronisation mechanism, which is why compositing is a separate offline pass.
It also means the panel can be redesigned without rebooting Isaac Sim.

Layout is a fixed 1920x1400: the sim scaled into 1920x1080 on top, and a 320px
terminal band below carrying the transcript. No architecture diagram -- the band
is a scrolling log of labelled stage blocks, newest at the bottom.

PEFA gets one block per stage per step (Oracle -> Robot LLM Executor -> Judge),
condensed to the decisive line but never collapsing steps together. FOLBP returns
its whole plan in a single Oracle call, so ORACLE_PLAN clears the band and prints
the entire plan verbatim -- two columns when it is long -- and the Z3 and
Validator verdicts then stamp onto it in place.
"""
import argparse
import bisect
import json
import os
import re
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ appearance

W, H = 1920, 1400
SIM_H = 1080
BAND_H = H - SIM_H                      # 320

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
F_MONO = os.path.join(FONT_DIR, "DejaVuSansMono.ttf")
F_MONO_B = os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf")
F_SANS_B = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

BG = (13, 17, 23)
BAND_BG = (13, 17, 23)
RULE = (33, 38, 45)
FG = (201, 209, 217)
DIM = (110, 118, 129)
DIMMER = (72, 79, 88)

C_BLUE = (88, 166, 255)
C_AMBER = (210, 153, 34)
C_VIOLET = (163, 113, 247)
C_GREEN = (63, 185, 80)
C_RED = (248, 81, 73)
C_ORANGE = (240, 136, 62)
C_PINK = (219, 97, 162)
C_CYAN = (57, 197, 207)

PAD_X = 34
HEADER_H = 40
LINE_H = 23
BODY_TOP = SIM_H + HEADER_H + 8
MAX_LINES = (BAND_H - HEADER_H - 20) // LINE_H     # 12

CPS = 220.0          # typewriter characters per second
PLAN_DWELL_S = 5.0   # how long a full FOLBP plan holds the band
COL_W = 96           # wrap width, characters


class Theme:
    def __init__(self):
        self.mono = ImageFont.truetype(F_MONO, 17)
        self.mono_b = ImageFont.truetype(F_MONO_B, 17)
        self.small = ImageFont.truetype(F_MONO, 14)
        self.head = ImageFont.truetype(F_SANS_B, 19)
        self.head_dim = ImageFont.truetype(F_MONO, 15)
        # Monospace: one advance width serves for all span arithmetic.
        self.cw = self.mono.getbbox("M")[2] - self.mono.getbbox("M")[0]


# ------------------------------------------------------------------ inline spans

RE_ENTITY = re.compile(r"<[^>]+>\(\d+\)")
RE_VERB = re.compile(r"\[[a-z_]+\]")


def spans(text, base=FG):
    """Split a line into (text, colour) runs: entities cyan, [verbs] amber."""
    marks = []
    for m in RE_ENTITY.finditer(text):
        marks.append((m.start(), m.end(), C_CYAN))
    for m in RE_VERB.finditer(text):
        marks.append((m.start(), m.end(), C_AMBER))
    if not marks:
        return [(text, base)]
    marks.sort()
    out, cur = [], 0
    for a, b, col in marks:
        if a < cur:
            continue
        if a > cur:
            out.append((text[cur:a], base))
        out.append((text[a:b], col))
        cur = b
    if cur < len(text):
        out.append((text[cur:], base))
    return out


def wrap(text, width=COL_W):
    """Word-wrap, preserving explicit newlines."""
    lines = []
    for para in str(text).split("\n"):
        para = para.rstrip()
        if not para:
            continue
        while len(para) > width:
            cut = para.rfind(" ", 0, width)
            if cut <= 0:
                cut = width
            lines.append(para[:cut])
            para = para[cut:].lstrip()
        lines.append(para)
    return lines


# ------------------------------------------------------------------ blocks

class Block:
    """One transcript entry: a labelled header line plus body lines."""

    def __init__(self, t, label, accent, lines=None, note="", mark="",
                 kind="plain", cols=1):
        self.t = t            # wall clock, for deciding when it becomes visible
        self.vt = 0.0         # video time of first appearance, for animation
        self.label = label
        self.accent = accent
        self.lines = lines or []
        self.note = note          # right-aligned on the header (latency, counts)
        self.mark = mark          # "ok" / "bad" / ""
        self.kind = kind          # "plain" | "plan"
        self.cols = cols

    @property
    def height(self):
        if self.kind == "plan" and self.cols == 2:
            return 1 + (len(self.lines) + 1) // 2
        return 1 + len(self.lines)

    @property
    def chars(self):
        return sum(len(x) for x in self.lines) or 1


def _first(text):
    """The first non-empty line of a multi-line LLM response."""
    for ln in str(text).split("\n"):
        if ln.strip():
            return ln.strip()
    return ""


RE_GOAL = re.compile(r"(\w+)_(<[^>]+>\(\d+\))_(<[^>]+>\(\d+\))")


def _goals(blob):
    """Turn "['inside_<battery>(39)_<box>(52)']" into "<battery>(39) inside <box>(52)"."""
    found = RE_GOAL.findall(str(blob))
    if not found:
        return str(blob)
    return ",  ".join("%s %s %s" % (obj, rel, dest) for rel, obj, dest in found)


RE_AGENT_ECHO = re.compile(r"^\s*\w+_\d+\s*:\s*$")


def _bridge_info(body, ok):
    """The sim's success reply is "<agent>: " with an empty info -- not a message."""
    if not body or RE_AGENT_ECHO.match(str(body)):
        return "executed in simulation" if ok else "rejected by the simulator"
    return str(body)


def _sentence(text, limit=200):
    """First sentence-ish chunk, for condensing a verbose LLM answer."""
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return t
    cut = t.rfind(". ", 0, limit)
    return t[:cut + 1] if cut > 40 else t[:limit].rstrip() + "…"


# FOLBP tags whose text is already covered by a richer block, or is pure noise.
FOLBP_DROP = (
    "OPEN", "CLOSE", "INIT",
    "received plan with", "  plan[", "requesting full plan",
    "mode=pefa_llm", "mode=deterministic",
)


def blocks_folbp(events):
    """Condense FOLBP's [FOLBP][TAG] stream into transcript blocks.

    Everything routes through ArenaMP._log, so the tag vocabulary here is exactly
    the one in FOLBP/LLM_oracle.py: ORACLE / FOL / Z3 / R / VAL / EXEC / A / B /
    C / D / ENV / FINAL, plus the WS_* and ORACLE_PLAN emits added for the video.
    """
    out = []
    for e in events:
        stage, body, title = e["stage"], e.get("body", ""), e.get("title", "")
        t = e["t"]

        if stage in ("OPEN", "CLOSE", "INIT", "LLM_CALL"):
            continue
        if stage == "ORACLE" and any(k in body for k in FOLBP_DROP):
            continue
        if stage == "EXEC" and body.startswith("mode="):
            continue

        if stage == "TASK":
            # FOLBP emits this twice: once as a plain _log line ("goal: ...") and
            # once as the structured emit that carries `framework`. Keep only the
            # structured one, or the video opens on the same paragraph twice.
            if "framework" not in e:
                continue
            out.append(Block(t, "TASK", C_BLUE, wrap(body)))

        elif stage == "ITER":
            m = re.search(r"outer_iter=(\d+)", body)
            if m and m.group(1) == "1":
                continue                       # the first pass needs no announcement
            if m:
                out.append(Block(t, "RE-PLAN", C_ORANGE,
                                 ["oracle called again — iteration %s" % m.group(1)]))
            else:
                out.append(Block(t, "ITER", DIM, wrap(body)))

        elif stage == "ORACLE_PLAN":
            plan = e.get("plan", [])
            lines = ["%2d. %-22s %s" % (i + 1, s.get("agent", ""), s.get("action", ""))
                     for i, s in enumerate(plan)]
            out.append(Block(t, "ORACLE LLM", C_BLUE, lines,
                             note="%d steps, 1 call" % len(plan),
                             kind="plan", cols=2 if len(plan) > 6 else 1))

        elif stage == "FOL":
            m = re.search(r"unary=(\d+) binary=(\d+)", body)
            note = "%s unary · %s binary" % m.groups() if m else ""
            out.append(Block(t, "FOL GENERATOR", C_VIOLET,
                             ["grounded the scene graph into typed predicates"],
                             note=note))

        elif stage == "Z3":
            if body.startswith("SAT"):
                out.append(Block(t, "Z3 SMT", C_GREEN, [body], mark="ok"))
            elif body.startswith("UNSAT"):
                out.append(Block(t, "Z3 SMT", C_RED, wrap(body), mark="bad"))
            elif "verification disabled" in body:
                out.append(Block(t, "Z3 SMT", DIM, [body]))

        elif stage == "SHADOW":
            out.append(Block(t, "Z3 SHADOW", C_RED, wrap(body), mark="bad"))

        elif stage == "R":
            out.append(Block(t, "UNSAT REPAIR", C_ORANGE, wrap(body)))

        elif stage == "VAL":
            ok = body.startswith("accepted")
            out.append(Block(t, "PLAN VALIDATOR", C_GREEN if ok else C_RED,
                             wrap(body), mark="ok" if ok else "bad"))

        elif stage == "EXEC":
            m = re.match(r"step\[(\d+)\] (.+?) -> (.+)", body)
            if m:
                out.append(Block(t, "ROBOT EXECUTOR", C_AMBER,
                                 ["%s   %s" % (m.group(2), m.group(3))],
                                 note="plan step %d" % (int(m.group(1)) + 1)))
            else:
                out.append(Block(t, "ROBOT EXECUTOR", C_RED, wrap(body), mark="bad"))

        elif stage == "WS_SEND":
            out.append(Block(t, "→ SIMULATOR", C_CYAN, ["%s   %s" % (title, body)]))

        elif stage == "WS_RECV":
            ok = e.get("success", False)
            out.append(Block(t, "← SIMULATOR", C_GREEN if ok else C_RED,
                             [_bridge_info(body, ok)],
                             mark="ok" if ok else "bad"))

        elif stage == "ENV":
            m = re.search(r"step=(\d+).*?done=(\w+)", body)
            if m:
                done = m.group(2) == "True"
                rem = re.search(r"remaining_goals=(\[.*\])", body)
                line = ("goal reached" if done else
                        "still to satisfy:  %s" % (_goals(rem.group(1)) if rem else "?"))
                out.append(Block(t, "ENVIRONMENT", C_GREEN if done else DIM,
                                 [line], note="step %s" % m.group(1),
                                 mark="ok" if done else ""))

        elif stage in ("A", "B", "C", "D"):
            names = {"A": "TASK GRAPH", "B": "CONFLICT ANALYZER",
                     "C": "CDCL ENGINE", "D": "CONSTRAINT INJECTION"}
            if stage == "A" and "FAILED" not in body:
                continue                       # only surface the failures
            if stage == "D" and body.startswith("oracle_prefix"):
                continue
            out.append(Block(t, names[stage], C_PINK, wrap(_sentence(body, 180))))

        elif stage == "RESULT":
            ok = e.get("success", False)
            out.append(Block(t, "RESULT", C_GREEN if ok else C_RED,
                             [("TASK COMPLETE — " if ok else "TASK FAILED — ") + body],
                             note="%s LLM calls" % e.get("llm", "?"),
                             mark="ok" if ok else "bad"))
    return out


def blocks_pefa(events):
    """Condense PEFA's stream.

    PEFA is a per-step dialogue loop, so every step contributes its own Oracle ->
    Robot LLM Executor -> Judge trio. Those are never merged: the point of the
    video is watching each step get proposed, grounded and judged in turn.
    """
    out = []
    for e in events:
        stage, body, title = e["stage"], e.get("body", ""), e.get("title", "")
        t = e["t"]

        if stage in ("OPEN", "CLOSE", "LLM_CALL"):
            continue

        elif stage == "TASK":
            out.append(Block(t, "TASK", C_BLUE, wrap(body)))

        elif stage == "STEP":
            out.append(Block(t, "STEP %s" % e.get("step", "?"), DIM, [], note=""))

        elif stage == "ORACLE":
            out.append(Block(t, "ORACLE LLM", C_BLUE, wrap(_sentence(body, 260)),
                             note=e.get("note", "")))

        elif stage == "ORACLE_NORM":
            out.append(Block(t, "ORACLE → ROBOT", C_BLUE, wrap(_first(body))))

        elif stage == "ACTIONS":
            out.append(Block(t, "ACTION SET", DIM,
                             ["%s candidate action(s) enumerated from the current state"
                              % e.get("n", "?")]))

        elif stage == "EXECUTOR":
            out.append(Block(t, "ROBOT LLM EXECUTOR", C_AMBER,
                             wrap(_sentence(body, 220)),
                             note=title))

        elif stage == "EXECUTOR_PICK":
            ok = bool(body)
            out.append(Block(t, "EXECUTOR → ACTION", C_AMBER if ok else C_RED,
                             [body or "no action parsed"],
                             note=title, mark="ok" if ok else "bad"))

        elif stage == "JUDGE":
            out.append(Block(t, "JUDGE", C_VIOLET, wrap(_sentence(body, 220))))

        elif stage == "WS_SEND":
            out.append(Block(t, "→ SIMULATOR", C_CYAN, ["%s   %s" % (title, body)]))

        elif stage == "WS_RECV":
            ok = e.get("success", False)
            out.append(Block(t, "← SIMULATOR", C_GREEN if ok else C_RED,
                             [_bridge_info(body, ok)],
                             mark="ok" if ok else "bad"))

        elif stage == "ENV":
            done = e.get("done", False)
            out.append(Block(t, "ENVIRONMENT", C_GREEN if done else DIM,
                             [body], note="step %s" % e.get("step", "?"),
                             mark="ok" if done else ""))

        elif stage == "RESULT":
            ok = e.get("success", False)
            out.append(Block(t, "RESULT", C_GREEN if ok else C_RED,
                             [("TASK COMPLETE — " if ok else "TASK FAILED — ") + body],
                             note="%s LLM calls" % e.get("llm", "?"),
                             mark="ok" if ok else "bad"))
    return out


# ------------------------------------------------------------------ band

class Band:
    def __init__(self, theme, framework, header):
        self.th = theme
        self.framework = framework.upper()
        self.header = header

    def draw(self, img, blocks, t, tv, t0, stats):
        """`t` is wall clock (what has happened yet); `tv` is video time.

        The two diverge sharply under native pacing, where one output frame is
        ~0.2 s of wall clock. Animation has to run on video time or the plan block
        would hold for well under a second on screen.
        """
        d = ImageDraw.Draw(img)
        d.rectangle([0, SIM_H, W, H], fill=BAND_BG)
        d.line([(0, SIM_H), (W, SIM_H)], fill=RULE, width=2)

        self._header(d, t, t0, stats)

        visible = [b for b in blocks if b.t <= t]
        if not visible:
            return

        # A full FOLBP plan owns the band for a beat after it arrives: it is the
        # thing worth reading, and the ask was to show it whole rather than
        # condensed. Checking "newest block" is not enough -- the FOL/Z3/Validator
        # verdicts land within milliseconds of the plan and would evict it
        # instantly. So hold on the most recent plan while it is still fresh.
        plans = [b for b in visible if b.kind == "plan"]
        if plans and (tv - plans[-1].vt) < PLAN_DWELL_S:
            self._block(d, plans[-1], BODY_TOP, tv, full=True)
            return

        # Otherwise fill upward from the bottom with the most recent blocks.
        chosen, used = [], 0
        for b in reversed(visible):
            if used + b.height > MAX_LINES:
                break
            chosen.append(b)
            used += b.height
        chosen.reverse()

        y = BODY_TOP + (MAX_LINES - used) * LINE_H
        for i, b in enumerate(chosen):
            newest = (i == len(chosen) - 1)
            fade = 1.0 if i >= len(chosen) - 2 else 0.55
            y = self._block(d, b, y, tv, fade=fade, typing=newest)

    def _header(self, d, t, t0, stats):
        y = SIM_H + 11
        d.text((PAD_X, y), self.framework, font=self.th.head, fill=C_BLUE)
        x = PAD_X + d.textlength(self.framework, font=self.th.head) + 16
        bits = [self.header]
        if stats.get("step"):
            bits.append("step %s" % stats["step"])
        if stats.get("calls"):
            bits.append("%s LLM calls" % stats["calls"])
        bits.append("%02d:%02d" % (int(t - t0) // 60, int(t - t0) % 60))
        d.text((x, y + 3), "  ·  ".join(bits), font=self.th.head_dim, fill=DIM)
        d.line([(PAD_X, SIM_H + HEADER_H - 4), (W - PAD_X, SIM_H + HEADER_H - 4)],
               fill=RULE, width=1)

    def _block(self, d, b, y, tv, fade=1.0, full=False, typing=False):
        th = self.th

        def mix(c):
            return tuple(int(BAND_BG[i] + (c[i] - BAND_BG[i]) * fade) for i in range(3))

        # header line: ▸ LABEL ............................ note  ✓
        d.text((PAD_X, y), "▸", font=th.mono_b, fill=mix(b.accent))
        d.text((PAD_X + 20, y), b.label, font=th.mono_b, fill=mix(b.accent))
        if b.note:
            nx = W - PAD_X - d.textlength(b.note, font=th.small) - (26 if b.mark else 0)
            d.text((nx, y + 3), b.note, font=th.small, fill=mix(DIM))
        if b.mark:
            d.text((W - PAD_X - 18, y), "✓" if b.mark == "ok" else "✗",
                   font=th.mono_b, fill=mix(C_GREEN if b.mark == "ok" else C_RED))
        y += LINE_H

        # Typewriter, on the newest block only. Several stages can land in the same
        # millisecond (FOL/Z3/Validator all do), and letting them all type at once
        # reads as a glitch rather than as output arriving.
        reveal = 10 ** 9
        if typing or full:
            reveal = int(max(0.0, tv - b.vt) * CPS)

        if b.kind == "plan" and b.cols == 2 and not full:
            rows = (len(b.lines) + 1) // 2
            for r in range(rows):
                for c in range(2):
                    k = r + c * rows
                    if k >= len(b.lines):
                        continue
                    self._line(d, b.lines[k], PAD_X + 24 + c * (W // 2 - 40),
                               y + r * LINE_H, mix, reveal)
            return y + rows * LINE_H

        for ln in b.lines:
            if reveal <= 0:
                break
            self._line(d, ln, PAD_X + 24, y, mix, reveal)
            reveal -= len(ln)
            y += LINE_H
        return y

    def _line(self, d, text, x, y, mix, reveal):
        if reveal < len(text):
            text = text[:max(0, reveal)]
        for run, col in spans(text):
            d.text((x, y), run, font=self.th.mono, fill=mix(col))
            x += len(run) * self.th.cw


# ------------------------------------------------------------------ time mapping

def motion_spans(events):
    """The stretches where a robot is actually moving, from the bridge traffic.

    WS_SEND is the action arriving at the simulator; the matching WS_RECV is the
    simulator reporting it finished. Everything between the two is a robot walking
    or flying. Everything outside is the planner thinking, with the scene static.

    Exact markers, so no heuristic about "quiet stretches" is needed.
    """
    spans, opened = [], None
    for e in sorted(events, key=lambda e: e["t"]):
        if e["stage"] == "WS_SEND":
            opened = e["t"]
        elif e["stage"] == "WS_RECV" and opened is not None:
            spans.append((opened, e["t"]))
            opened = None
    return spans


def build_adaptive(t_start, t_end, spans, fps, motion_speed, think_speed):
    """Fast through the motion, near real time through the reasoning.

    The two phases want opposite things. A robot crossing a room is a minute of
    walking that says nothing the viewer cannot see in ten seconds. An Oracle call
    is four seconds of wall clock carrying a nine-step plan that needs time on
    screen to read. Playing both at one rate has to fail one of them.
    """
    out, t = [], t_start
    dt = 1.0 / fps
    i, n = 0, len(spans)
    while t < t_end:
        out.append(t)
        while i < n and spans[i][1] < t:      # spans are sorted and disjoint
            i += 1
        moving = i < n and spans[i][0] <= t <= spans[i][1]
        t += dt * (motion_speed if moving else think_speed)
    return out


def build_timeline(ftimes, t_start, t_end, events, fps, pace,
                   timelapse, quiet_s, speed,
                   motion_speed=8.0, think_speed=1.5):
    """The real timestamp each output frame should show.

    Two pacings, and the difference is large because the simulator does not
    render anywhere near real time -- the RTX renderer manages about 5 fps, so a
    run that takes eight minutes of wall clock is only ~2400 captured frames.

    native (default)
        One output frame per captured sim frame. The video then runs at exactly
        the pace of sim.mp4: every frame the simulator actually drew, played at
        `fps`. An eight-minute run becomes 80 seconds, and the motion is smooth
        because no frame is held or repeated.

    realtime
        One output frame per 1/fps of wall clock. True to the clock, but it holds
        each sim frame for ~6 output frames, so the motion is a 5 fps judder
        stretched over the full run length.
    """
    if pace == "adaptive":
        spans = motion_spans(events)
        if not spans:
            print("[compose] no bridge traffic in the log — falling back to native")
        else:
            moving = sum(b - a for a, b in spans)
            print("[compose] %d motion span(s), %.0fs moving / %.0fs thinking; "
                  "playing motion at %.1fx and reasoning at %.1fx"
                  % (len(spans), moving, (t_end - t_start) - moving,
                     motion_speed, think_speed))
            return build_adaptive(t_start, t_end, spans, fps,
                                  motion_speed, think_speed)

    if pace in ("native", "adaptive"):
        times = [t for t in ftimes if t_start <= t <= t_end]
        if timelapse:
            times = _thin_quiet(times, events, quiet_s, speed)
        return times

    if not timelapse:
        n = int((t_end - t_start) * fps)
        return [t_start + i / float(fps) for i in range(n)]

    marks = sorted(e["t"] for e in events)
    out, t = [], t_start
    dt = 1.0 / fps
    while t < t_end:
        out.append(t)
        nxt = bisect.bisect_right(marks, t)
        gap = (marks[nxt] - t) if nxt < len(marks) else (t_end - t)
        # Inside a quiet stretch, and not about to run into the next event.
        t += dt * (speed if gap > quiet_s else 1.0)
    return out


def _thin_quiet(times, events, quiet_s, speed):
    """Drop frames inside stretches where no stage fires, keeping 1 in `speed`."""
    marks = sorted(e["t"] for e in events)
    out, skipped = [], 0
    for t in times:
        nxt = bisect.bisect_right(marks, t)
        gap = (marks[nxt] - t) if nxt < len(marks) else 0.0
        if gap > quiet_s and skipped < int(speed) - 1:
            skipped += 1
            continue
        skipped = 0
        out.append(t)
    return out


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compose(run_dir, framework, out_path, fps, pace, timelapse, quiet_s, speed,
            limit=None, crf=20, motion_speed=8.0, think_speed=1.5):
    sim_mp4 = os.path.join(run_dir, "sim.mp4")
    frames = [r for r in load_jsonl(os.path.join(run_dir, "frames.jsonl"))
              if "mark" not in r]
    events = load_jsonl(os.path.join(run_dir, "events.jsonl"))
    if not frames:
        sys.exit("no frames in %s" % run_dir)
    if not events:
        sys.exit("no events in %s" % run_dir)

    ftimes = [r["t"] for r in frames]
    blocks = (blocks_folbp if framework == "folbp" else blocks_pefa)(events)
    print("[compose] %d sim frames, %d events -> %d transcript blocks"
          % (len(frames), len(events), len(blocks)))

    # Start when both streams are live, so the video does not open on a blank band.
    t0 = max(ftimes[0], events[0]["t"])
    t_end = min(ftimes[-1], events[-1]["t"] + 4.0)
    if t_end <= t0:
        sys.exit("sim capture and event log do not overlap in time — "
                 "were they from the same run?")

    # Prefer the structured TASK emit (the one carrying `framework`); FOLBP's
    # plain _log('TASK', ...) lines are debug text, not a caption.
    task = next((e.get("body") or e.get("title", "")
                 for e in events
                 if e["stage"] == "TASK" and "framework" in e), "")
    if not task:
        task = next((e.get("title") or e.get("body", "")
                     for e in events if e["stage"] == "TASK"), "")
    theme = Theme()
    if len(task) > 80:
        task = task[:80].rsplit(" ", 1)[0] + "…"
    band = Band(theme, framework, task)

    times = build_timeline(ftimes, t0, t_end, events, fps, pace,
                           timelapse, quiet_s, speed,
                           motion_speed, think_speed)
    if limit:
        times = times[:limit]
    if not times:
        sys.exit("timeline is empty — no sim frames inside the event window")

    # Stamp each block with the video time it first appears, so the typewriter and
    # the plan dwell run on screen seconds rather than wall-clock seconds.
    for b in blocks:
        i = bisect.bisect_left(times, b.t)
        b.vt = min(i, len(times) - 1) / float(fps)

    print("[compose] pace=%s -> %d output frames (%.0fs at %d fps) from %.0fs of run"
          % (pace, len(times), len(times) / float(fps), fps, t_end - t0))

    reader = imageio.get_reader(sim_mp4)
    # CRF rather than imageio's `quality`: at 1920x1400 quality=9 lands around
    # 13 Mb/s, which makes a 3-minute demo a 260 MB file nobody wants to send.
    # CRF 20 is visually equivalent here (the sim half is already h264-encoded
    # once) at roughly a fifth the size. yuv420p is explicit because anything
    # else fails to play in browsers and most desktop players.
    # pixelformat= rather than an output_params -pix_fmt: imageio passes its own,
    # and duplicating it makes ffmpeg warn on every run.
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", macro_block_size=None,
        pixelformat="yuv420p",
        output_params=["-crf", str(crf), "-preset", "medium"],
    )

    # Both streams are monotonic, so walk the decoder forward rather than seeking.
    cur_idx, cur_frame = -1, None
    canvas = Image.new("RGB", (W, H), BG)

    try:
        for n, t in enumerate(times):
            want = bisect.bisect_right(ftimes, t) - 1
            want = max(0, min(want, len(frames) - 1))
            while cur_idx < want:
                try:
                    cur_frame = reader.get_next_data()
                    cur_idx += 1
                except (StopIteration, IndexError):
                    break
            if cur_frame is not None:
                im = Image.fromarray(cur_frame)
                if im.size != (W, SIM_H):
                    im = im.resize((W, SIM_H), Image.LANCZOS)
                canvas.paste(im, (0, 0))

            stats = current_stats(events, t)
            band.draw(canvas, blocks, t, n / float(fps), t0, stats)
            writer.append_data(np.asarray(canvas))

            if n % 200 == 0:
                print("[compose] %d/%d" % (n, len(times)), flush=True)
    finally:
        reader.close()
        writer.close()
    print("[compose] wrote %s" % out_path)


def current_stats(events, t):
    """Header counters as of time `t`."""
    step, calls = None, 0
    for e in events:
        if e["t"] > t:
            break
        # LLM_CALL comes from bench/llm_meter.py, one per physical API call.
        # Counting stage lines instead would overstate it by an order of magnitude.
        if e["stage"] == "LLM_CALL":
            calls += 1
        m = re.search(r"step=(\d+)", str(e.get("body", "")))
        if e["stage"] == "ENV" and m:
            step = m.group(1)
        elif e["stage"] == "ENV" and e.get("step"):
            step = e["step"]
    return {"step": step, "calls": calls or None}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True,
                    help="Directory holding sim.mp4, frames.jsonl and events.jsonl.")
    ap.add_argument("--framework", required=True, choices=["pefa", "folbp"],
                    help="Selects the stage vocabulary and the plan-block behaviour.")
    ap.add_argument("--out", default=None, help="Output mp4. Default <run_dir>/final.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pace", default="adaptive",
                    choices=["adaptive", "native", "realtime"],
                    help="adaptive (default): run fast through robot motion and "
                         "near real time through the reasoning, using the bridge's "
                         "send/receive markers to tell them apart. native: one "
                         "output frame per captured sim frame -- sim.mp4's own "
                         "pace, uniform. realtime: true wall-clock duration.")
    ap.add_argument("--motion_speed", type=float, default=8.0,
                    help="Playback multiplier while a robot is moving. Default 8x.")
    ap.add_argument("--think_speed", type=float, default=1.5,
                    help="Playback multiplier while the planner is reasoning. "
                         "Default 1.5x -- slow enough to read the panel.")
    ap.add_argument("--timelapse", action="store_true",
                    help="Additionally speed up stretches where no stage fires.")
    ap.add_argument("--quiet_s", type=float, default=8.0,
                    help="A gap longer than this counts as quiet. Default 8s.")
    ap.add_argument("--speed", type=float, default=3.5,
                    help="Playback multiplier inside a quiet stretch. Default 3.5x.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Render only the first N frames (for design iteration).")
    ap.add_argument("--crf", type=int, default=20,
                    help="x264 quality, lower is better/bigger. Default 20.")
    a = ap.parse_args()

    out = a.out or os.path.join(a.run_dir, "final.mp4")
    compose(a.run_dir, a.framework, out, a.fps, a.pace, a.timelapse, a.quiet_s,
            a.speed, a.limit, a.crf, a.motion_speed, a.think_speed)


if __name__ == "__main__":
    main()
