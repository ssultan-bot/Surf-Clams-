"""Pick measurement lines by hand, one image after another, and save each.

A local OpenCV window -- no browser. By default it walks the reader-comparison
folder (the images that have human-read ages), advancing to the next image as
soon as you save one.

Four stages per image, each confirmed by you:

    1  SAM prompts   left-click = positive (on the shell)
                     right-click = negative (debris / scale bar / background)
    2  pick mask     SAM offers several candidates; page through and choose,
                     or go back and add points if none is good
    3  endpoints     click the two ends, then tune the upward shift
    4  save          only on your explicit keypress -> next image

All on-screen text is ASCII: cv2.putText renders only Hershey fonts, so any
non-ASCII character comes out as '?'.

Saves the layout the analysis step expects:
    results/<image>_<tag>_<time>/{line.csv, mask.png, mask.npy,
                                  overlay.png, overlay_on_photo.png,
                                  shell_cut.png, meta.json}

Needs the env that has torch + segment_anything:

    D:\\Anaconda\\envs\\sea\\python.exe pick_line.py
    D:\\Anaconda\\envs\\sea\\python.exe pick_line.py --skip-done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from surfclam import imaging, edge, refine

WIN = "surfclam - pick a line"
RESULTS = config.PROJECT_DIR / "results"
RESULTS.mkdir(exist_ok=True)
SAM_CKPT = config.PROJECT_DIR / "models" / "sam_vit_h_4b8939.pth"
HQ_CKPT = config.PROJECT_DIR / "models" / "sam_hq_vit_h.pth"
DEFAULT_DIR = (config.DATA_DIR / "Surf Clam hinge images" / "NOAA Surf Clam Survey"
               / "Reader comparison selected images")
GPU_PY = r"D:\Anaconda\envs\sea\python.exe"

KEY_ENTER, KEY_ESC = 13, 27
HELP = {
    1: "[1/4] Lclick=positive  Rclick=negative  u=undo  c=clear  ENTER=run SAM  . ,=next/prev img  ESC=quit",
    2: "[2/4] n/p=browse candidates  g=edge-snap on/off  ENTER=use this one  b=back and add points",
    3: "[3/4] click 2 ends | +/- up/down 10  ]/[ up/down 50  t=lower/upper  r=repick  b=back  ENTER=ok",
    4: "[4/4] s=SAVE and go to next image    b=back and keep adjusting    ESC=quit",
}

# The SAM model is expensive to load, so it is kept for the whole session and
# only the per-image embedding is recomputed when the image changes.
# Both models are held at once (ViT-H ~2.4GB + SAM2 hiera-L ~0.9GB fits easily on
# a 24GB card) so a single image can be shown segmented by each, side by side in
# the candidate list, and the choice made per image rather than per session.
_MODELS: dict[str, dict] = {}
_DEVICE = None


def require_gpu_env(kinds):
    """Fail early and usefully if this interpreter cannot run the chosen models."""
    need = {"torch"}
    if "sam" in kinds:
        need.add("segment_anything")
    if "sam2" in kinds:
        need.add("sam2")
    if "hq" in kinds:
        need.add("segment_anything_hq")
    missing = [m for m in sorted(need) if __import__("importlib").util.find_spec(m) is None]
    if missing:
        raise SystemExit(
            f"\nMissing module(s): {', '.join(missing)}\n"
            f"You ran: {sys.executable}\n"
            f"Use the env that has them:\n\n    {GPU_PY} {Path(__file__).name}\n"
        )
    import torch
    global _DEVICE
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if _DEVICE == "cpu":
        print(f"  ! torch {torch.__version__} reports no CUDA -- running on CPU, "
              f"expect ~1 min per image instead of seconds")


def get_predictor(kind: str):
    """Load a model once and keep it for the session."""
    if kind not in _MODELS:
        if kind == "sam2":
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large",
                                                      device=_DEVICE)
        else:
            if kind == "hq":
                from segment_anything_hq import sam_model_registry, SamPredictor
                ck = HQ_CKPT
            else:
                from segment_anything import sam_model_registry, SamPredictor
                ck = SAM_CKPT
            if not ck.exists():
                raise SystemExit(f"checkpoint not found: {ck}")
            pred = SamPredictor(sam_model_registry["vit_h"](checkpoint=str(ck)).to(_DEVICE))
        _MODELS[kind] = {"predictor": pred, "key": None}
    return _MODELS[kind]


def drop_specks(mask: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """Remove connected components smaller than `frac` of the biggest one.

    SAM 2 in particular returns the shell plus a scatter of stray specks (11-13
    components where SAM ViT-H gives 3); they are never wanted and they make the
    candidate hard to judge on screen.
    """
    m = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= frac * areas.max())[0]
    return np.isin(lab, keep).astype(np.uint8)


def already_done(stem: str) -> bool:
    return any(p.is_dir() and p.name.startswith(stem + "_") and (p / "line.csv").exists()
               for p in RESULTS.iterdir())


# ---------------------------------------------------------------------------
class Picker:
    def __init__(self, files: list[Path], tag: str, work: int, disp_w: int, disp_h: int,
                 model_kinds: list[str] | None = None):
        self.model_kinds = model_kinds or ["sam", "sam2"]
        self.files = files
        self.tag = tag
        self.work = work
        self.disp_w, self.disp_h = disp_w, disp_h
        self.fi = -1
        self.load(0)

    # -- per-image state ---------------------------------------------------
    def load(self, i: int):
        self.fi = i % len(self.files)
        self.path = str(self.files[self.fi])
        self.bgr = imaging.load_bgr(self.path)
        self.h, self.w = self.bgr.shape[:2]
        self.scale = min(1.0, self.disp_w / self.w, self.disp_h / self.h)
        self.stage = 1
        self.pts: list[tuple[float, float, int]] = []
        self.masks: list[np.ndarray] = []
        self.scores: list[float] = []
        self.labels: list[str] = []
        self.cand = 0
        self.snap = True                 # boundary snapping, toggled with 'g'
        self._snap_cache: dict[int, np.ndarray] = {}
        self._gray = imaging.to_gray(self.bgr)
        self._grad = None
        self.mask: np.ndarray | None = None
        self.ends: list[tuple[float, float]] = []
        self.dy = 0.0
        self.side_lower = True
        self.line: np.ndarray | None = None
        self.line_alt: np.ndarray | None = None      # the other edge, drawn dim
        self.chosen = ""                             # which candidate / refinement was used
        done = "  [already done]" if already_done(Path(self.path).stem) else ""
        self.msg = f"click positive points on the shell, negative on junk{done}"
        print(f"\n--- {self.fi+1}/{len(self.files)}  {Path(self.path).name}  "
              f"({self.w}x{self.h}){done}")

    def next_image(self):
        self.load(self.fi + 1)

    def prev_image(self):
        self.load(self.fi - 1)

    # -- helpers -----------------------------------------------------------
    def to_full(self, x, y):
        return x / self.scale, y / self.scale

    def show(self, img):
        d = cv2.resize(img, (int(self.w * self.scale), int(self.h * self.scale)))
        bar = np.zeros((66, d.shape[1], 3), np.uint8)
        head = f"[{self.fi+1}/{len(self.files)}] {Path(self.path).name}"
        cv2.putText(bar, head, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 120), 1)
        cv2.putText(bar, HELP[self.stage], (12, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(bar, self.msg, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1)
        cv2.imshow(WIN, np.vstack([d, bar]))

    # -- rendering ---------------------------------------------------------
    def render(self):
        if self.stage == 1:
            v = self.bgr.copy()
            r = max(10, self.w // 150)
            for x, y, lb in self.pts:
                c = (0, 220, 0) if lb == 1 else (0, 0, 235)
                cv2.circle(v, (int(x), int(y)), r, c, -1)
                cv2.circle(v, (int(x), int(y)), r, (255, 255, 255), 3)
            return v
        if self.stage == 2:
            # one candidate at a time, at full width -- stacking all of them into
            # one frame squashed each to a third of its height and hid the detail
            # you actually need in order to judge the mask.
            if not self.masks:
                return self.bgr.copy()
            i = self.cand % len(self.masks)
            m, s = self.current_mask(), self.scores[i]
            v = self.bgr.copy()
            t = np.zeros_like(v); t[m > 0] = (60, 60, 255)
            v = cv2.addWeighted(v, 1.0, t, 0.42, 0)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cnts, -1, (0, 255, 0), 4)
            pieces = cv2.connectedComponents(m, 8)[0] - 1
            cv2.putText(v, f"{i+1}/{len(self.masks)} [{self.labels[i]}]  score={s:.3f}  "
                           f"area={100*m.sum()/(self.h*self.w):.1f}%  pieces={pieces}  "
                           f"snap={'ON' if self.snap else 'off'}",
                        (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 5)
            return v
        v = self.bgr.copy()
        v[self.mask == 0] = 0
        # the edge you are NOT using, dim -- so you can compare and press t
        if self.line_alt is not None:
            cv2.polylines(v, [np.round(self.line_alt).astype(np.int32)], False,
                          (110, 110, 110), 4, cv2.LINE_AA)
        for i, (x, y) in enumerate(self.ends):
            cv2.circle(v, (int(x), int(y)), 20, (0, 0, 255) if i == 0 else (0, 255, 0), -1)
        if self.line is not None:
            cv2.polylines(v, [np.round(self.line).astype(np.int32)], False,
                          (0, 255, 255), 6, cv2.LINE_AA)
            arrow = "UP" if self.dy > 0 else ("DOWN" if self.dy < 0 else "on edge")
            cv2.putText(v, f"{'LOWER' if self.side_lower else 'UPPER'} edge   "
                           f"shift {self.dy:+.0f}px {arrow}   (t=switch  0=reset)",
                        (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 255), 5)
        return v

    # -- actions -----------------------------------------------------------
    def run_sam(self):
        if not any(p[2] == 1 for p in self.pts):
            self.msg = "need at least one positive point"
            return
        coords = np.array([[p[0], p[1]] for p in self.pts], dtype=np.float32)
        labels = np.array([p[2] for p in self.pts], dtype=np.int32)
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        masks, scores, labels_ = [], [], []

        for kind in self.model_kinds:
            self.msg = f"running {kind} ..."
            self.show(self.render()); cv2.waitKey(1)
            slot = get_predictor(kind)
            pred = slot["predictor"]
            import torch
            if slot["key"] != self.path:              # embedding is the slow part
                with torch.inference_mode():
                    pred.set_image(rgb)
                slot["key"] = self.path

            if kind == "sam2":
                with torch.inference_mode():
                    m, sc, _ = pred.predict(point_coords=coords, point_labels=labels,
                                            multimask_output=True)
                for j, i in enumerate(np.argsort(sc)[::-1]):
                    masks.append(np.asarray(m[i]).astype(np.uint8))
                    scores.append(float(sc[i]))
                    labels_.append(f"sam2-{j+1}")
            else:
                m, sc, lg = pred.predict(point_coords=coords, point_labels=labels,
                                         multimask_output=True)
                order = list(np.argsort(sc)[::-1])
                for j, i in enumerate(order):
                    masks.append(m[i].astype(np.uint8))
                    scores.append(float(sc[i]))
                    labels_.append(f"{kind}-raw{j+1}")
                # Second pass seeded with each candidate's own logits: little
                # change on an already good mask, but it can rescue a badly
                # broken one (a 131-piece background blob became a coherent
                # shell), so refined versions are offered as extra choices.
                for j, i in enumerate(order):
                    mr, sr, _ = pred.predict(point_coords=coords, point_labels=labels,
                                             mask_input=lg[i][None], multimask_output=False)
                    masks.append(mr[0].astype(np.uint8))
                    scores.append(float(sr[0]))
                    labels_.append(f"{kind}-ref{j+1}")

        self.masks, self.scores, self.labels = masks, scores, labels_
        self.cand = 0
        self._snap_cache = {}
        self.stage = 2
        self.msg = (f"{_DEVICE}: {len(self.masks)} candidates from "
                    f"{'+'.join(self.model_kinds)} - n/p to browse, ENTER to accept")

    def current_mask(self) -> np.ndarray:
        """The candidate on screen, with boundary snapping applied if enabled."""
        i = self.cand % len(self.masks)
        key = (i, self.snap)
        if key not in self._snap_cache:
            m = drop_specks(self.masks[i])
            if self.snap:
                if self._grad is None:
                    self._grad = refine.edge_strength(self._gray)
                m = refine.snap_to_edges(self._gray, m, grad=self._grad)
            self._snap_cache[key] = m
        return self._snap_cache[key]

    def recompute(self):
        """Build both edges. Which one carries the readable rings differs from
        shell to shell, so the unchosen one is kept and drawn dim: seeing the two
        side by side is what makes the choice obvious."""
        self.line = self.line_alt = None
        if self.mask is None or len(self.ends) < 2:
            return
        f = min(1.0, self.work / max(self.h, self.w))
        small = cv2.resize(self.mask, (int(self.w * f), int(self.h * f)),
                           interpolation=cv2.INTER_NEAREST) * 255
        a = (self.ends[0][0] * f, self.ends[0][1] * f)
        b = (self.ends[1][0] * f, self.ends[1][1] * f)
        cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            self.msg = "empty mask - press b and pick another candidate"
            return
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        ia = int(np.argmin(np.hypot(cnt[:, 0] - a[0], cnt[:, 1] - a[1])))
        ib = int(np.argmin(np.hypot(cnt[:, 0] - b[0], cnt[:, 1] - b[1])))
        if ia > ib:
            ia, ib = ib, ia
        arc1, arc2 = cnt[ia:ib + 1], np.vstack([cnt[ib:], cnt[:ia + 1]])
        lower_is_1 = arc1[:, 1].mean() > arc2[:, 1].mean()
        lower_arc = arc1 if lower_is_1 else arc2
        upper_arc = arc2 if lower_is_1 else arc1

        def build(arc):
            if len(arc) < 4:
                return None
            if np.hypot(*(arc[0] - np.array(a))) > np.hypot(*(arc[-1] - np.array(a))):
                arc = arc[::-1]
            return edge.offset_up(edge.smooth_axis(arc, tol=2.5) / f, self.dy)

        lo, up = build(lower_arc), build(upper_arc)
        self.line = lo if self.side_lower else up
        self.line_alt = up if self.side_lower else lo
        if self.line is None:
            self.msg = "endpoints too close - press r and re-pick"
            return
        length = float(np.sum(np.hypot(*np.diff(self.line, axis=0).T)))
        alt_len = (float(np.sum(np.hypot(*np.diff(self.line_alt, axis=0).T)))
                   if self.line_alt is not None else 0.0)
        if self.dy > 0:
            shift = f"shift +{self.dy:.0f}px UP"
        elif self.dy < 0:
            shift = f"shift {self.dy:.0f}px DOWN"
        else:
            shift = "shift 0 (on the edge)"
        self.msg = (f"EDGE={'LOWER' if self.side_lower else 'UPPER'} (t=switch) | "
                    f"{shift} | length {length:.0f}px (other edge {alt_len:.0f}px)")

    def save(self) -> Path:
        stem = Path(self.path).stem
        tag = f"_{self.tag}" if self.tag else ""
        folder = RESULTS / f"{stem}{tag}_{time.strftime('%m%d_%H%M%S')}"
        folder.mkdir(parents=True, exist_ok=True)
        cut = self.bgr.copy(); cut[self.mask == 0] = 0
        np.savetxt(folder / "line.csv", self.line, delimiter=",",
                   header="x,y", comments="", fmt="%.2f")
        cv2.imwrite(str(folder / "mask.png"), (self.mask > 0).astype(np.uint8) * 255)
        np.save(folder / "mask.npy", (self.mask > 0).astype(np.uint8))
        cv2.imwrite(str(folder / "shell_cut.png"), cut)
        cv2.imwrite(str(folder / "overlay.png"), self.render())
        onp = self.bgr.copy()
        cv2.polylines(onp, [np.round(self.line).astype(np.int32)], False,
                      (0, 255, 255), 6, cv2.LINE_AA)
        cv2.imwrite(str(folder / "overlay_on_photo.png"), onp)
        meta = {
            "image": str(self.path),
            "sam_points": [{"x": p[0], "y": p[1], "label": int(p[2])} for p in self.pts],
            "endpoints": [list(self.ends[0]), list(self.ends[1])],
            "side": "lower" if self.side_lower else "upper",
            "offset_up_px": float(self.dy),
            "work_scale_px": int(self.work),
            "line_points": int(len(self.line)),
            "line_length_px": float(np.sum(np.hypot(*np.diff(self.line, axis=0).T))),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mask_variant": self.chosen,
            "picked_by": "manual",
        }
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        return folder

    # -- mouse -------------------------------------------------------------
    def on_mouse(self, ev, x, y, flags, param):
        if y >= int(self.h * self.scale):           # clicked the help bar
            return
        fx, fy = self.to_full(x, y)
        if self.stage == 1:
            if ev == cv2.EVENT_LBUTTONDOWN:
                self.pts.append((fx, fy, 1))
            elif ev == cv2.EVENT_RBUTTONDOWN:
                self.pts.append((fx, fy, 0))
            else:
                return
            npos = sum(1 for p in self.pts if p[2] == 1)
            self.msg = f"positive {npos}, negative {len(self.pts)-npos}  -  ENTER to run SAM"
        elif self.stage == 3 and ev == cv2.EVENT_LBUTTONDOWN:
            if len(self.ends) >= 2:
                self.ends = []
                self.line = None
            self.ends.append((fx, fy))
            if len(self.ends) == 2:
                self.recompute()
            else:
                self.msg = "P1 set - now click P2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="folder of images to walk")
    ap.add_argument("--image", default=None, help="single image (ignores --dir)")
    ap.add_argument("--skip-done", action="store_true", help="skip images already saved")
    ap.add_argument("--models", default="sam,sam2",
                    help="which segmenters to offer candidates from, comma separated: "
                         "sam (ViT-H), sam2 (SAM 2.1 hiera-L), hq (SAM-HQ)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--work", type=int, default=1600)
    ap.add_argument("--display-width", type=int, default=1500)
    ap.add_argument("--display-height", type=int, default=850)
    args = ap.parse_args()

    kinds = [k.strip() for k in args.models.split(",") if k.strip()]
    bad = [k for k in kinds if k not in ("sam", "sam2", "hq")]
    if bad:
        raise SystemExit(f"unknown model(s): {bad}. choose from sam, sam2, hq")
    require_gpu_env(kinds)

    if args.image:
        files = [Path(args.image)]
    else:
        d = Path(args.dir)
        files = sorted([p for p in d.glob("*.jpg") if not p.name.startswith("._")])
        if not files:
            raise SystemExit(f"no jpg in {d}")
        n_all = len(files)
        if args.skip_done:
            files = [p for p in files if not already_done(p.stem)]
            print(f"{d.name}: {n_all} images, {n_all - len(files)} already done, "
                  f"{len(files)} to go")
            if not files:
                raise SystemExit("All done.")
        else:
            done = sum(1 for p in files if already_done(p.stem))
            print(f"{d.name}: {n_all} images ({done} already saved; "
                  f"use --skip-done to skip them)")

    print(f"segmenters: {' + '.join(kinds)}")
    p = Picker(files, args.tag, args.work, args.display_width, args.display_height, kinds)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, p.on_mouse)

    while True:
        p.show(p.render())
        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        if k == KEY_ESC:
            break

        # image navigation is available whenever we are not mid-line
        if p.stage in (1, 2) and k == ord("."):
            p.next_image(); continue
        if p.stage in (1, 2) and k == ord(","):
            p.prev_image(); continue

        if p.stage == 1:
            if k == ord("u") and p.pts:
                p.pts.pop(); p.msg = f"undo - {len(p.pts)} points left"
            elif k == ord("c"):
                p.pts.clear(); p.msg = "cleared"
            elif k == KEY_ENTER:
                p.run_sam()

        elif p.stage == 2:
            if k == ord("n"):
                p.cand = (p.cand + 1) % len(p.masks); p.msg = ""
            elif k == ord("p"):
                p.cand = (p.cand - 1) % len(p.masks); p.msg = ""
            elif k == ord("g"):
                p.snap = not p.snap
                p.msg = f"edge-snap {'ON' if p.snap else 'off'}"
            elif ord("1") <= k <= ord("9"):
                i = k - ord("1")
                if i < len(p.masks):
                    p.cand = i; p.msg = f"showing {p.labels[i]} - ENTER to accept"
            elif k == KEY_ENTER:
                i = p.cand % len(p.masks)
                p.mask = p.current_mask()
                p.chosen = f"{p.labels[i]}{'+snap' if p.snap else ''}"
                p.stage = 3
                p.ends = []; p.line = None
                p.msg = f"using {p.chosen} - click the two endpoints"
            elif k == ord("b"):
                p.stage = 1
                p.msg = "add points (especially negative ones), then ENTER to re-run SAM"

        elif p.stage == 3:
            # dy may go negative: the useful direction is *into* the shell, which
            # is downward when the upper edge is the one being traced.
            if k in (ord("+"), ord("=")):
                p.dy += 10; p.recompute()
            elif k in (ord("-"), ord("_")):
                p.dy -= 10; p.recompute()
            elif k == ord("]"):
                p.dy += 50; p.recompute()
            elif k == ord("["):
                p.dy -= 50; p.recompute()
            elif k == ord("0"):
                p.dy = 0.0; p.recompute()
            elif k == ord("t"):
                p.side_lower = not p.side_lower; p.recompute()
            elif k == ord("r"):
                p.ends = []; p.line = None; p.msg = "re-pick the two endpoints"
            elif k == ord("b"):
                p.stage = 2; p.msg = "back to candidate choice"
            elif k == KEY_ENTER and p.line is not None:
                p.stage = 4
                p.msg = "happy with it? s = save and next image, b = keep adjusting"

        elif p.stage == 4:
            if k == ord("s"):
                folder = p.save()
                print(f"  saved {folder.name}")
                if p.fi + 1 >= len(p.files):
                    print("\nAll images processed.")
                    p.msg = "last image - press ESC to quit"
                    p.stage = 3
                else:
                    p.next_image()
            elif k == ord("b"):
                p.stage = 3

    cv2.destroyAllWindows()
    print("\nDone. Results are in results/")


if __name__ == "__main__":
    main()
