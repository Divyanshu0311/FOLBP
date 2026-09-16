#!/usr/bin/env python3
"""Two-pane comparison segments for the ICRA submission video.

    compare_video.py sim  --left <folbp run_dir> --right <pefa run_dir> --out seg.mp4
    compare_video.py hw   --cues tools/hw_cues.json --out seg.mp4
    compare_video.py card --spec tools/cards/title.json --out card.mp4

`compose_video.py` films one run with its transcript underneath. This films two
runs against each other: the sim panes side by side, and underneath them a band
that is the whole argument of the paper -- FOLBP's plan is on screen complete
from the first Oracle call and ticks green as it executes, while PEFA's fills in
one line at a time with the call counter climbing behind it.

Palette, fonts and the entity/verb colouring come from compose_video so the two
demo styles stay siblings; only the layout is new.

Pacing is uniform and identical on both sides. Both house runs captured frames
at a steady ~4.55 fps, so one output frame per `--speed` seconds of wall clock
keeps the two durations in their true ratio (496.0s vs 331.7s = 1.50x). That
matters: the adaptive pacing compose_video uses by default compresses motion 8x
and reasoning 1.5x, which side by side would read as a 4.5x speedup and
overstate the result. The honest gap here is 1 LLM call against 55.
"""
import argparse
import bisect
import json
import math
import os
import re
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_video as cv          # palette, fonts, spans(), load_jsonl()

# ------------------------------------------------------------------ layout

W, H = 1920, 1080
FPS = 30

MARGIN = 14
GUTTER = 20
PANE_W = 936
PANE_H = 527                        # 936 * 720/1280, rounded to even

HEAD_H = 84                         # title strip
LABEL_H = 42                        # per-side name strip
PANE_Y = HEAD_H + LABEL_H           # 126
BAND_Y = PANE_Y + PANE_H            # 653
CAP_H = 92                          # caption strip at the foot
CAP_Y = H - CAP_H                   # 988

LEFT_X = MARGIN
RIGHT_X = MARGIN + PANE_W + GUTTER

BG = cv.BG
RULE = cv.RULE
FG = cv.FG
DIM = cv.DIM
DIMMER = cv.DIMMER

C_OURS = cv.C_GREEN                 # FOLBP accent
C_BASE = cv.C_ORANGE                # PEFA accent


class Fonts:
    """Bigger than compose_video's Theme -- this band is read at arm's length."""

    def __init__(self):
        d = cv.FONT_DIR
        j = os.path.join
        self.title = ImageFont.truetype(j(d, "DejaVuSans-Bold.ttf"), 30)
        self.sub = ImageFont.truetype(j(d, "DejaVuSans.ttf"), 20)
        self.name = ImageFont.truetype(j(d, "DejaVuSans-Bold.ttf"), 24)
        self.name_sub = ImageFont.truetype(j(d, "DejaVuSansMono.ttf"), 17)
        self.huge = ImageFont.truetype(j(d, "DejaVuSans-Bold.ttf"), 52)
        self.stat = ImageFont.truetype(j(d, "DejaVuSans-Bold.ttf"), 15)
        self.mono = ImageFont.truetype(j(d, "DejaVuSansMono.ttf"), 17)
        self.mono_b = ImageFont.truetype(j(d, "DejaVuSansMono-Bold.ttf"), 17)
        self.chip = ImageFont.truetype(j(d, "DejaVuSansMono-Bold.ttf"), 16)
        self.cap = ImageFont.truetype(j(d, "DejaVuSans.ttf"), 26)
        self.card_h = ImageFont.truetype(j(d, "DejaVuSans-Bold.ttf"), 64)
        self.card_b = ImageFont.truetype(j(d, "DejaVuSans.ttf"), 30)
        self.card_s = ImageFont.truetype(j(d, "DejaVuSansMono.ttf"), 22)


def draw_spans(dr, xy, text, font, base=FG):
    """compose_video.spans() rendered run by run.

    Advance on font.getlength(), not on a per-character width derived from an
    ink bounding box -- the box is narrower than the advance, and over a long
    line the colour runs creep left until words collide.
    """
    x, y = xy
    for run, col in cv.spans(text, base=base):
        dr.text((x, y), run, font=font, fill=col)
        x += font.getlength(run)
    return x


def chrome(img, dr, f, title, sub, left, right, caption):
    """Everything that does not depend on the current frame's state."""
    dr.rectangle([0, 0, W, H], fill=BG)

    dr.text((MARGIN + 6, 18), title, font=f.title, fill=FG)
    if sub:
        dr.text((MARGIN + 8, 54), sub, font=f.sub, fill=DIM)
    dr.line([(0, HEAD_H - 1), (W, HEAD_H - 1)], fill=RULE)

    for x, side, accent in ((LEFT_X, left, C_OURS), (RIGHT_X, right, C_BASE)):
        dr.rectangle([x, HEAD_H + 6, x + 5, HEAD_H + LABEL_H - 10], fill=accent)
        dr.text((x + 16, HEAD_H + 7), side["label"], font=f.name, fill=accent)
        wpx = dr.textlength(side["label"], font=f.name)
        dr.text((x + 26 + wpx, HEAD_H + 14), side["sub"], font=f.name_sub, fill=DIM)

    dr.line([(0, BAND_Y), (W, BAND_Y)], fill=RULE)
    dr.line([(RIGHT_X - GUTTER // 2, BAND_Y + 10),
             (RIGHT_X - GUTTER // 2, CAP_Y - 10)], fill=RULE)
    dr.line([(0, CAP_Y), (W, CAP_Y)], fill=RULE)

    if caption:
        wpx = dr.textlength(caption, font=f.cap)
        dr.text(((W - wpx) / 2, CAP_Y + 30), caption, font=f.cap, fill=FG)


def stats_row(dr, f, x, items):
    """`LABEL  value` cells across the top of a band column."""
    y = BAND_Y + 8
    for label, value, col in items:
        dr.text((x, y), label, font=f.stat, fill=DIMMER)
        dr.text((x, y + 14), value, font=f.huge, fill=col)
        x += max(dr.textlength(value, font=f.huge) + 60,
                 dr.textlength(label, font=f.stat) + 60)


def checklist(dr, f, x, rows, done, running, ghost=False):
    """Two columns of plan steps: green tick done, amber caret running, dim ahead.

    `ghost` dims rows that have not been proposed yet -- PEFA's list does not
    exist in advance, so its unfilled rows stay blank rather than pending.
    """
    y0 = BAND_Y + 124
    col_w = 470
    per_col = 6
    for i, row in enumerate(rows):
        cx = x + (i // per_col) * col_w
        cy = y0 + (i % per_col) * 24
        if i < done:
            dr.text((cx, cy), cv.MARK_DONE if hasattr(cv, "MARK_DONE") else "✓",
                    font=f.mono_b, fill=cv.C_GREEN)
        elif i == running:
            dr.text((cx, cy), "▶", font=f.mono_b, fill=cv.C_AMBER)
        elif ghost:
            continue
        else:
            dr.text((cx, cy), "·", font=f.mono, fill=DIMMER)
        base = FG if i <= running else DIMMER
        if i < done:
            base = DIM
        draw_spans(dr, (cx + 22, cy), row, f.mono, base=base)


def chip(dr, f, x, y, text, col):
    wpx = dr.textlength(text, font=f.chip)
    dr.rectangle([x, y, x + wpx + 22, y + 28], outline=col)
    dr.text((x + 11, y + 6), text, font=f.chip, fill=col)
    return x + wpx + 32


def paste(canvas, frame, x):
    if frame is None:
        return
    im = Image.fromarray(frame)
    if im.size != (PANE_W, PANE_H):
        im = im.resize((PANE_W, PANE_H), Image.LANCZOS)
    canvas.paste(im, (x, PANE_Y))


def caption_at(script, t):
    out = ""
    for start, text in script:
        if t >= start:
            out = text
    return out


# ------------------------------------------------------------------ sim source

class SimSide:
    """One run: its sim frames, its events, and the state derived from them."""

    def __init__(self, run_dir, framework, label, sub):
        self.run_dir = run_dir
        self.framework = framework
        self.label = label
        self.sub = sub
        frames = [r for r in cv.load_jsonl(os.path.join(run_dir, "frames.jsonl"))
                  if "mark" not in r]
        self.events = cv.load_jsonl(os.path.join(run_dir, "events.jsonl"))
        self.ftimes = [r["t"] for r in frames]
        self.t0 = max(self.ftimes[0], self.events[0]["t"])
        self.t_end = self.ftimes[-1]
        # The run ends when the planner stops, not when the recorder does: the
        # capture keeps rolling a few seconds past the last event, and counting
        # those would put a number on screen the paper does not report.
        self.t_run_end = self.events[-1]["t"]
        self.span = self.t_run_end - self.t0
        self.reader = imageio.get_reader(os.path.join(run_dir, "sim.mp4"))
        self.cur_idx, self.cur_frame = -1, None

        # The executed plan, in order. EXEC (FOLBP) is the certified plan; PEFA
        # has no plan, so its rows are the actions its executor actually picked,
        # each stamped with the time it was chosen.
        self.rows, self.row_t = [], []
        if framework == "folbp":
            for e in self.events:
                m = re.match(r"step\[(\d+)\] (.+?) -> (.+)$", str(e.get("body", "")))
                if e["stage"] == "EXEC" and m:
                    self.rows.append(m.group(3))
                    self.row_t.append(self.t_plan())
        else:
            for e in self.events:
                if e["stage"] == "EXECUTOR_PICK":
                    self.rows.append(str(e.get("body", "")))
                    self.row_t.append(e["t"])

        self.call_t = [e["t"] for e in self.events if e["stage"] == "LLM_CALL"]
        self.env_t = [e["t"] for e in self.events if e["stage"] == "ENV"]

    def t_plan(self):
        """When the certified plan became available (FOLBP only)."""
        for e in self.events:
            if e["stage"] == "VAL":
                return e["t"]
        return self.t0

    def frame_at(self, t):
        want = bisect.bisect_right(self.ftimes, t) - 1
        want = max(0, min(want, len(self.ftimes) - 1))
        while self.cur_idx < want:
            try:
                self.cur_frame = self.reader.get_next_data()
                self.cur_idx += 1
            except (StopIteration, IndexError):
                break
        return self.cur_frame

    def state(self, t):
        done = bisect.bisect_right(self.env_t, t)
        return {
            "calls": bisect.bisect_right(self.call_t, t),
            "done": done,
            "shown": bisect.bisect_right(self.row_t, t),
            "running": done if done < len(self.rows) else -1,
            "elapsed": max(0.0, min(t, self.t_run_end) - self.t0),
            "finished": t >= self.t_run_end or done >= len(self.rows),
        }

    def phase(self, t):
        """PEFA's current leg of Oracle -> Executor -> Judge."""
        last = None
        for e in self.events:
            if e["t"] > t:
                break
            if e["stage"] == "LLM_CALL":
                last = e.get("title", "")
        return last


def band_folbp(dr, f, x, side, st, t):
    stats_row(dr, f, x + 6, [
        ("LLM CALLS", str(st["calls"]), C_OURS),
        ("STEPS", "%d/%d" % (st["done"], len(side.rows)), FG),
        ("ELAPSED", "%.0f s" % st["elapsed"], DIM),
    ])
    y = BAND_Y + 88
    if st["shown"] == 0:
        chip(dr, f, x + 6, y, "ORACLE — one call for the whole plan", cv.C_BLUE)
    elif st["finished"]:
        chip(dr, f, x + 6, y, "✓ 11/11 STEPS · GOAL REACHED", cv.C_GREEN)
    else:
        nx = chip(dr, f, x + 6, y, "Z3 ✓ SAT", cv.C_GREEN)
        nx = chip(dr, f, nx, y, "VALIDATOR ✓", cv.C_GREEN)
        chip(dr, f, nx, y, "0 further LLM calls", DIM)
    checklist(dr, f, x + 6, side.rows, st["done"], st["running"])


def band_pefa(dr, f, x, side, st, t):
    stats_row(dr, f, x + 6, [
        ("LLM CALLS", str(st["calls"]), C_BASE),
        ("STEPS", "%d/%d" % (st["done"], len(side.rows)), FG),
        ("ELAPSED", "%.0f s" % st["elapsed"], DIM),
    ])
    y = BAND_Y + 88
    ph = side.phase(t)
    if st["finished"]:
        chip(dr, f, x + 6, y, "✓ 11/11 STEPS · GOAL REACHED", cv.C_GREEN)
    else:
        nx = x + 6
        for name, key, col in (("ORACLE", "oracle", cv.C_BLUE),
                               ("EXECUTOR", "executor", cv.C_VIOLET),
                               ("JUDGE", "judge", cv.C_PINK)):
            lit = (ph == key)
            nx = chip(dr, f, nx, y, name, col if lit else DIMMER)
        chip(dr, f, nx, y, "5 calls per step", DIM)
    checklist(dr, f, x + 6, side.rows, st["done"], st["running"], ghost=True)


def render_sim(left, right, out_path, speed, tail_s, script, title, sub, crf,
               limit=None):
    f = Fonts()
    span = max(left.span, right.span)
    # Ceil, not trunc: truncating leaves the last output frame a fraction of a
    # second short of the longer run's end, so that side never reaches its own
    # completion state and holds at n-1 steps through the whole tail.
    n_frames = int(math.ceil(span / speed * FPS)) + 1
    tail = int(tail_s * FPS)
    print("[compare] %.0fs vs %.0fs of run at %.2fx -> %d frames (%.0fs + %.0fs hold)"
          % (left.span, right.span, speed, n_frames + tail,
             n_frames / float(FPS), tail_s))

    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264", macro_block_size=None,
        pixelformat="yuv420p",
        output_params=["-crf", str(crf), "-preset", "medium"])
    canvas = Image.new("RGB", (W, H), BG)
    total = min(n_frames + tail, limit) if limit else n_frames + tail
    try:
        for n in range(total):
            tau = min(n, n_frames - 1) * speed / float(FPS)
            dr = ImageDraw.Draw(canvas)
            chrome(canvas, dr, f, title, sub,
                   {"label": left.label, "sub": left.sub},
                   {"label": right.label, "sub": right.sub},
                   caption_at(script, n / float(FPS)))
            for side, x, band in ((left, LEFT_X, band_folbp),
                                  (right, RIGHT_X, band_pefa)):
                t = side.t0 + tau
                paste(canvas, side.frame_at(t), x)
                band(dr, f, x, side, side.state(t), t)
            writer.append_data(np.asarray(canvas))
            if n % 150 == 0:
                print("[compare] %d/%d" % (n, total), flush=True)
    finally:
        left.reader.close()
        right.reader.close()
        writer.close()
    print("[compare] wrote %s" % out_path)


# ------------------------------------------------------------------ hw source

class HwSide:
    """Hand-recorded footage plus the cue times scrubbed out of it.

    No frames.jsonl, no events.jsonl -- the camera never saw the planner. The
    video is the clock, exactly as hw_overlay.py assumes, so the step boundaries
    here were read off the footage rather than derived from the transcripts.
    """

    def __init__(self, spec, steps):
        self.label = spec["label"]
        self.sub = spec["sub"]
        self.cues = [float(c) for c in spec["cues"]]
        self.end = float(spec["end"])
        self.span = self.end
        self.steps = steps
        self.reader = imageio.get_reader(spec["video"])
        self.cur_idx, self.cur_frame = -1, None
        self.src_fps = self.reader.get_meta_data()["fps"]

    def frame_at(self, t):
        want = int(round(t * self.src_fps))
        while self.cur_idx < want:
            try:
                self.cur_frame = self.reader.get_next_data()
                self.cur_idx += 1
            except (StopIteration, IndexError):
                break
        return self.cur_frame

    def state(self, t):
        started = bisect.bisect_right(self.cues, t)
        finished = t >= self.end
        return {
            "done": started if finished else max(0, started - 1),
            "running": -1 if finished else started - 1,
            "elapsed": min(t, self.end),
            "finished": finished,
        }


def band_hw(dr, f, x, side, st, accent):
    stats_row(dr, f, x + 6, [
        ("STEPS", "%d/%d" % (st["done"], len(side.steps)), FG),
        ("ELAPSED", "%.0f s" % st["elapsed"], accent),
    ])
    y = BAND_Y + 88
    if st["finished"]:
        chip(dr, f, x + 6, y, "✓ BASKET DOWN AT SETPOINT A — SUCCESS", cv.C_GREEN)
    elif st["running"] >= 4:
        chip(dr, f, x + 6, y, "DRONE", cv.C_VIOLET)
    else:
        chip(dr, f, x + 6, y, "TURTLEBOT3 + OpenMANIPULATOR-X", cv.C_CYAN)

    y0 = BAND_Y + 128
    for i, (act, gloss) in enumerate(side.steps):
        cy = y0 + i * 28
        if i < st["done"]:
            dr.text((x + 6, cy), "✓", font=f.mono_b, fill=cv.C_GREEN)
            base = DIM
        elif i == st["running"]:
            dr.text((x + 6, cy), "▶", font=f.mono_b, fill=cv.C_AMBER)
            base = FG
        else:
            dr.text((x + 6, cy), "·", font=f.mono, fill=DIMMER)
            base = DIMMER
        nx = draw_spans(dr, (x + 28, cy), act, f.mono, base=base)
        dr.text((nx + 14, cy + 1), gloss, font=f.mono,
                fill=DIM if i == st["running"] else DIMMER)


def render_hw(spec, out_path, speed, tail_s, crf, limit=None):
    f = Fonts()
    steps = [tuple(s) for s in spec["steps"]]
    left = HwSide(spec["left"], steps)
    right = HwSide(spec["right"], steps)
    script = [(float(a), b) for a, b in spec.get("captions", [])]
    span = max(left.span, right.span)
    # Ceil, not trunc: truncating leaves the last output frame a fraction of a
    # second short of the longer run's end, so that side never reaches its own
    # completion state and holds at n-1 steps through the whole tail.
    n_frames = int(math.ceil(span / speed * FPS)) + 1
    tail = int(tail_s * FPS)
    print("[compare] hw %.0fs vs %.0fs at %.2fx -> %d frames"
          % (left.span, right.span, speed, n_frames + tail))

    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264", macro_block_size=None,
        pixelformat="yuv420p",
        output_params=["-crf", str(crf), "-preset", "medium"])
    canvas = Image.new("RGB", (W, H), BG)
    total = min(n_frames + tail, limit) if limit else n_frames + tail
    try:
        for n in range(total):
            tau = min(n, n_frames - 1) * speed / float(FPS)
            dr = ImageDraw.Draw(canvas)
            chrome(canvas, dr, f, spec["title"], spec.get("sub", ""),
                   {"label": left.label, "sub": left.sub},
                   {"label": right.label, "sub": right.sub},
                   caption_at(script, n / float(FPS)))
            for side, x, accent in ((left, LEFT_X, C_OURS),
                                    (right, RIGHT_X, C_BASE)):
                paste(canvas, side.frame_at(min(tau, side.end)), x)
                band_hw(dr, f, x, side, side.state(tau), accent)
            writer.append_data(np.asarray(canvas))
            if n % 150 == 0:
                print("[compare] %d/%d" % (n, total), flush=True)
    finally:
        left.reader.close()
        right.reader.close()
        writer.close()
    print("[compare] wrote %s" % out_path)


# ------------------------------------------------------------------ cards

def render_card(spec, out_path, crf):
    """A still title/section/closing card, drawn with the same type as the bands
    so the cuts between them do not look like two different videos."""
    f = Fonts()
    canvas = Image.new("RGB", (W, H), BG)
    if spec.get("backdrop"):
        bd = Image.open(spec["backdrop"]["path"]).convert("RGB")
        scale = max(W / float(bd.size[0]), H / float(bd.size[1]))
        bd = bd.resize((int(bd.size[0] * scale) + 1, int(bd.size[1] * scale) + 1),
                       Image.LANCZOS).crop((0, 0, W, H))
        canvas = Image.blend(bd, Image.new("RGB", (W, H), BG),
                             spec["backdrop"].get("dim", 0.82))
    dr = ImageDraw.Draw(canvas)

    block = (46 if spec.get("eyebrow") else 0) \
        + 78 * len(spec.get("heading", [])) \
        + (18 + 44 * len(spec.get("body", [])) if spec.get("body") else 0) \
        + (14 + 34 * len(spec.get("mono", [])) if spec.get("mono") else 0)
    y = int(spec.get("top", max(120, (H - block) // 2)))
    if spec.get("eyebrow"):
        dr.text((200, y), spec["eyebrow"], font=f.card_s, fill=cv.C_GREEN)
        y += 46
    for line in spec.get("heading", []):
        dr.text((200, y), line, font=f.card_h, fill=FG)
        y += 78
    y += 18
    for line in spec.get("body", []):
        dr.text((202, y), line, font=f.card_b, fill=DIM)
        y += 44
    y += 14
    for line in spec.get("mono", []):
        draw_spans(dr, (202, y), line, f.card_s, base=DIM)
        y += 34
    if spec.get("note"):
        dr.line([(200, H - 150), (W - 200, H - 150)], fill=RULE)
        dr.text((202, H - 132), spec["note"], font=f.card_s, fill=DIMMER)

    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264", macro_block_size=None,
        pixelformat="yuv420p",
        output_params=["-crf", str(crf), "-preset", "medium"])
    arr = np.asarray(canvas)
    try:
        for _ in range(int(float(spec.get("seconds", 5)) * FPS)):
            writer.append_data(arr)
    finally:
        writer.close()
    print("[compare] wrote %s (%.1fs)" % (out_path, float(spec.get("seconds", 5))))


# ------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sim", "hw", "card"])
    ap.add_argument("--left", help="sim: FOLBP run dir")
    ap.add_argument("--right", help="sim: PEFA run dir")
    ap.add_argument("--cues", help="hw: cue/caption spec json")
    ap.add_argument("--spec", help="card: card spec json")
    ap.add_argument("--script", help="sim: caption script json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=9.92,
                    help="wall-clock seconds per output second, same on both sides")
    ap.add_argument("--tail", type=float, default=4.0,
                    help="seconds to hold the final frame")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--limit", type=int, default=None,
                    help="render only the first N frames, for checking layout")
    a = ap.parse_args()

    if a.mode == "sim":
        script = json.load(open(a.script)) if a.script else {}
        left = SimSide(a.left, "folbp", script.get("left_label", "FOLBP (ours)"),
                       script.get("left_sub", ""))
        right = SimSide(a.right, "pefa", script.get("right_label", "PEFA"),
                        script.get("right_sub", ""))
        render_sim(left, right, a.out, a.speed, a.tail,
                   [(float(x), y) for x, y in script.get("captions", [])],
                   script.get("title", ""), script.get("sub", ""), a.crf, a.limit)
    elif a.mode == "hw":
        render_hw(json.load(open(a.cues)), a.out, a.speed, a.tail, a.crf, a.limit)
    else:
        render_card(json.load(open(a.spec)), a.out, a.crf)


if __name__ == "__main__":
    main()
