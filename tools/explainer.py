#!/usr/bin/env python3
"""Animated architecture slides for the ICRA video.

    explainer.py --spec tools/icra/slide_pipeline.json --out slide.mp4

A slide is a declarative diagram: boxes, arrows between them, and a time at
which each one appears. Nothing is laid out automatically -- every coordinate is
in the spec -- because these three slides are drawn once and read a hundred
times, and an auto-router would spend its cleverness fighting the one layout
that actually reads well.

Chrome (title strip, caption strip, palette, fonts) is shared with
compare_video.py so a cut from a slide into a comparison segment does not look
like a cut into a different video.

Spec:

    {"seconds": 18,
     "title": "...", "sub": "...",
     "nodes": {"id": {"x":,"y":,"w":,"h":, "label":"", "note":["",""],
                      "kind":"llm|sym|env|io|learn|bad", "at": 2.0}},
     "edges": [{"from":"a","to":"b","at":2.5,"label":"plan P","kind":"sym"}],
     "chips": [{"x":,"y":,"text":"","kind":"","at":}],
     "captions": [[0.0,"..."]]}

An edge connects two node ids by their facing sides, inferred from their
geometry; `path` overrides that with explicit points when the connection has to
go around something (the feedback loops).
"""
import argparse
import json
import math
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_video as cv
import compare_video as cmp

W, H = cmp.W, cmp.H
FPS = cmp.FPS
HEAD_H = cmp.HEAD_H
CAP_Y = cmp.CAP_Y

KIND = {
    "llm":   cv.C_BLUE,
    "sym":   cv.C_GREEN,
    "env":   cv.C_VIOLET,
    "io":    cv.DIM,
    "learn": cv.C_ORANGE,
    "bad":   cv.C_RED,
    "note":  cv.C_AMBER,
}

FADE_S = 0.35


def lerp(c, t):
    """Fade a colour up from the background. t in [0,1]."""
    return tuple(int(cv.BG[i] + (c[i] - cv.BG[i]) * t) for i in range(3))


def alpha(at, t):
    if t < at:
        return 0.0
    return min(1.0, (t - at) / FADE_S)


class Fonts(cmp.Fonts):
    def __init__(self):
        super().__init__()
        d = cv.FONT_DIR
        j = os.path.join
        self.box = ImageFont.truetype(j(d, "DejaVuSans-Bold.ttf"), 23)
        self.box_s = ImageFont.truetype(j(d, "DejaVuSansMono.ttf"), 15)
        self.edge = ImageFont.truetype(j(d, "DejaVuSansMono.ttf"), 15)
        self.big = ImageFont.truetype(j(d, "DejaVuSans-Bold.ttf"), 46)


def sides(n):
    return {"l": (n["x"], n["y"] + n["h"] / 2),
            "r": (n["x"] + n["w"], n["y"] + n["h"] / 2),
            "t": (n["x"] + n["w"] / 2, n["y"]),
            "b": (n["x"] + n["w"] / 2, n["y"] + n["h"])}


def route(a, b):
    """Pick the facing sides of two boxes from their relative position."""
    ax, ay = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bx, by = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    if abs(bx - ax) >= abs(by - ay):
        return (sides(a)["r"], sides(b)["l"]) if bx > ax \
            else (sides(a)["l"], sides(b)["r"])
    return (sides(a)["b"], sides(b)["t"]) if by > ay \
        else (sides(a)["t"], sides(b)["b"])


def arrowhead(dr, p, q, col, size=11):
    """Filled triangle at q, pointing along p->q."""
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    pts = [q,
           (q[0] - size * math.cos(ang - 0.42), q[1] - size * math.sin(ang - 0.42)),
           (q[0] - size * math.cos(ang + 0.42), q[1] - size * math.sin(ang + 0.42))]
    dr.polygon(pts, fill=col)


def draw_node(dr, f, n, t):
    a = alpha(n.get("at", 0), t)
    if a <= 0:
        return
    col = lerp(KIND.get(n.get("kind", "sym"), cv.C_GREEN), a)
    body = lerp(cv.FG, a)
    note = lerp(cv.DIM, a)
    dr.rounded_rectangle([n["x"], n["y"], n["x"] + n["w"], n["y"] + n["h"]],
                         radius=9, outline=col, width=2)
    lines = n.get("note", [])
    ly = n["y"] + (n["h"] - (28 + 19 * len(lines))) / 2
    dr.text((n["x"] + 18, ly), n["label"], font=f.box, fill=body)
    for i, line in enumerate(lines):
        dr.text((n["x"] + 18, ly + 30 + 19 * i), line, font=f.box_s, fill=note)


def draw_edge(dr, f, e, nodes, t):
    a = alpha(e.get("at", 0), t)
    if a <= 0:
        return
    col = lerp(KIND.get(e.get("kind", "sym"), cv.DIM), a)
    if "path" in e:
        pts = [tuple(p) for p in e["path"]]
    else:
        p, q = route(nodes[e["from"]], nodes[e["to"]])
        pts = [p, q]
    dr.line(pts, fill=col, width=2)
    arrowhead(dr, pts[-2], pts[-1], col)
    if e.get("label"):
        lx, ly = e.get("label_at", (0, 0))
        mid = pts[len(pts) // 2]
        x = lx or mid[0] + 10
        y = ly or mid[1] - 26
        for i, line in enumerate(str(e["label"]).split("\n")):
            dr.text((x, y + 19 * i), line, font=f.edge, fill=col)


def draw_chip(dr, f, c, t):
    a = alpha(c.get("at", 0), t)
    if a <= 0:
        return
    col = lerp(KIND.get(c.get("kind", "io"), cv.DIM), a)
    font = f.big if c.get("big") else (f.card_s if c.get("plain") else f.chip)
    if c.get("big") or c.get("plain"):
        dr.text((c["x"], c["y"]), c["text"], font=font, fill=col)
    else:
        wpx = dr.textlength(c["text"], font=font)
        dr.rectangle([c["x"], c["y"], c["x"] + wpx + 22, c["y"] + 28], outline=col)
        dr.text((c["x"] + 11, c["y"] + 6), c["text"], font=font, fill=col)


class Figure:
    """A paper figure on a white panel, revealed region by region.

    The figure is the one in the paper -- redrawing it for the video would
    invite the two to drift. Unrevealed parts are washed out rather than hidden,
    so the shape of the whole pipeline is visible from the first frame and the
    spotlight only says where to look.
    """

    def __init__(self, spec):
        self.x, self.y, self.w = spec["x"], spec["y"], spec["w"]
        src = Image.open(spec["path"]).convert("RGB")
        self.h = int(round(self.w * src.size[1] / src.size[0]))
        self.full = src.resize((self.w, self.h), Image.LANCZOS)
        wash = Image.new("RGB", self.full.size, (255, 255, 255))
        self.dim = Image.blend(self.full, wash, spec.get("wash", 0.80))
        self.pad = spec.get("pad", 22)

    def draw(self, canvas, dr, spots, t):
        dr.rounded_rectangle(
            [self.x - self.pad, self.y - self.pad,
             self.x + self.w + self.pad, self.y + self.h + self.pad],
            radius=14, fill=(255, 255, 255))
        canvas.paste(self.dim, (self.x, self.y))
        for sp in spots:
            a = alpha(sp.get("at", 0), t)
            if a <= 0:
                continue
            x0, y0, x1, y1 = sp["rect"]
            box = (int(x0 * self.w), int(y0 * self.h),
                   int(x1 * self.w), int(y1 * self.h))
            crop = Image.blend(self.dim.crop(box), self.full.crop(box), a)
            canvas.paste(crop, (self.x + box[0], self.y + box[1]))
            # The accent ring flares as the region lands, then settles back so
            # a slide that has revealed six regions is not six shouting boxes.
            age = t - sp.get("at", 0)
            ring = max(0.0, 1.0 - age / 1.4)
            if ring > 0.02:
                col = KIND.get(sp.get("kind", "sym"), cv.C_GREEN)
                dr.rounded_rectangle(
                    [self.x + box[0] - 4, self.y + box[1] - 4,
                     self.x + box[2] + 4, self.y + box[3] + 4],
                    radius=10,
                    outline=tuple(int(255 + (c - 255) * ring) for c in col),
                    width=3)


def render(spec, out_path, crf, limit=None):
    f = Fonts()
    fig = Figure(spec["image"]) if spec.get("image") else None
    spots = spec.get("spots", [])
    nodes = spec.get("nodes", {})
    edges = spec.get("edges", [])
    chips = spec.get("chips", [])
    script = [(float(a), b) for a, b in spec.get("captions", [])]
    n_frames = int(math.ceil(float(spec["seconds"]) * FPS))
    if limit:
        n_frames = min(n_frames, limit)

    writer = imageio.get_writer(
        out_path, fps=FPS, codec="libx264", macro_block_size=None,
        pixelformat="yuv420p",
        output_params=["-crf", str(crf), "-preset", "medium"])
    try:
        for n in range(n_frames):
            t = n / float(FPS)
            canvas = Image.new("RGB", (W, H), cv.BG)
            dr = ImageDraw.Draw(canvas)

            dr.text((cmp.MARGIN + 6, 18), spec["title"], font=f.title, fill=cv.FG)
            if spec.get("sub"):
                dr.text((cmp.MARGIN + 8, 54), spec["sub"], font=f.sub, fill=cv.DIM)
            dr.line([(0, HEAD_H - 1), (W, HEAD_H - 1)], fill=cv.RULE)
            dr.line([(0, CAP_Y), (W, CAP_Y)], fill=cv.RULE)

            if fig:
                fig.draw(canvas, dr, spots, t)
            for e in edges:
                draw_edge(dr, f, e, nodes, t)
            for node in nodes.values():
                draw_node(dr, f, node, t)
            for c in chips:
                draw_chip(dr, f, c, t)

            cap = cmp.caption_at(script, t)
            if cap:
                wpx = dr.textlength(cap, font=f.cap)
                dr.text(((W - wpx) / 2, CAP_Y + 30), cap, font=f.cap, fill=cv.FG)

            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    print("[explainer] wrote %s (%.1fs)" % (out_path, n_frames / float(FPS)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    render(json.load(open(a.spec)), a.out, a.crf, a.limit)


if __name__ == "__main__":
    main()
