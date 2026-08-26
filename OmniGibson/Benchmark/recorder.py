"""Viewport capture for the demo videos.

Grabs the RGB the viewer camera already renders and writes it to an mp4, plus a
JSONL manifest carrying a wall-clock timestamp per frame.

That manifest is the whole point. The planner runs in a different process under a
different interpreter (system python3.10 vs. Isaac Sim's 3.7) and writes its own
timestamped stage-event stream; `tools/compose_video.py` joins the two on
`time.time()` afterwards. Nothing here knows the planner exists.

Frames go straight to h264 rather than to PNGs: a 13-step demo is roughly 12k
frames, which is ~12 GB of PNG and ~100 MB of mp4. The quality is set high enough
that the compositor's second encode does not visibly compound.

`og.sim.viewer_camera` is a VisionSensor created with modalities="rgb"
(simulator.py:175), so the grab below is the same one CameraMover.get_image()
does in omnigibson/utils/ui_utils.py:319.
"""
from __future__ import print_function

import json
import os
import time

import imageio
import omnigibson as og


class ViewportRecorder:
    """Writes sim.mp4 + frames.jsonl into `out_dir`. Call capture() after og.sim.step()."""

    def __init__(self, out_dir, fps=30, every_n=1, quality=9):
        self.out_dir = out_dir
        self.fps = fps
        self.every_n = max(1, int(every_n))
        self.quality = quality

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.video_path = os.path.join(out_dir, "sim.mp4")
        self.manifest_path = os.path.join(out_dir, "frames.jsonl")

        # macro_block_size=None: imageio otherwise silently pads dimensions up to a
        # multiple of 16, which would shift every frame relative to the compositor's
        # layout arithmetic.
        self._writer = imageio.get_writer(
            self.video_path, fps=fps, quality=quality, macro_block_size=None,
        )
        self._manifest = open(self.manifest_path, "w")
        self._n = 0
        self._steps = 0
        self._t_start = time.time()
        self._failures = 0

        print("[recorder] writing %s at %d fps (1 frame per %d sim step(s))"
              % (self.video_path, fps, self.every_n), flush=True)

    def capture(self):
        """Grab one frame if this sim step is due. Never raises."""
        self._steps += 1
        if self._steps % self.every_n:
            return
        try:
            rgb = og.sim.viewer_camera.get_obs()["rgb"][:, :, :3]
        except Exception as exc:
            # A capture failure must never take the simulation down; report the
            # first few and then stay quiet.
            self._failures += 1
            if self._failures <= 3:
                print("[recorder] frame grab failed (%s)" % exc, flush=True)
            return
        self._writer.append_data(rgb)
        self._manifest.write(json.dumps({"i": self._n, "t": time.time()}) + "\n")
        self._manifest.flush()
        self._n += 1

    def mark(self, label, **meta):
        """Note a named moment on the same clock (task start, action round, ...)."""
        rec = {"i": self._n, "t": time.time(), "mark": label}
        rec.update(meta)
        self._manifest.write(json.dumps(rec) + "\n")
        self._manifest.flush()

    def close(self):
        try:
            self._writer.close()
        except Exception:
            pass
        try:
            self._manifest.close()
        except Exception:
            pass
        elapsed = time.time() - self._t_start
        print("[recorder] %d frames over %.1fs -> %s"
              % (self._n, elapsed, self.video_path), flush=True)


class NullRecorder:
    """Stand-in when --record is off, so the sim loop needs no conditional."""

    def capture(self):
        pass

    def mark(self, label, **meta):
        pass

    def close(self):
        pass
