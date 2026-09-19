"""Interactive tool: click two points, get the edge line between them.

Works on the SAM-extracted shell (clean mask, no debris / scale bar / background),
so locating the margin is now a geometry problem rather than a segmentation one.

    D:\\Anaconda\\python.exe app.py
    # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

import config
from surfclam import imaging, edge

CUT = config.OUTPUT_DIR / "shell_cut.png"
MASK = config.OUTPUT_DIR / "shell_mask.png"

if not CUT.exists() or not MASK.exists():
    raise SystemExit(f"Run extract_shell.py first -- missing {CUT.name} / {MASK.name}")

BGR = cv2.imread(str(CUT))
MASK_FULL = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE)
RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
H, W = MASK_FULL.shape


def _marks(rgb, pts):
    out = rgb.copy()
    r = max(9, W // 170)
    for i, (x, y) in enumerate(pts):
        c = (255, 60, 60) if i == 0 else (60, 255, 60)
        cv2.circle(out, (int(x), int(y)), r, c, -1)
        cv2.putText(out, f"P{i+1}", (int(x) + r + 4, int(y) - r), cv2.FONT_HERSHEY_SIMPLEX,
                    r / 11, c, max(2, r // 5))
    return out


def _compute(pts, side, which, offset_pct, smooth_tol, show_on):
    f = min(1.0, side / max(H, W))
    small_mask = cv2.resize(MASK_FULL, (int(W * f), int(H * f)), interpolation=cv2.INTER_NEAREST)
    a = (pts[0][0] * f, pts[0][1] * f)
    b = (pts[1][0] * f, pts[1][1] * f)

    arc = edge.bottom_arc(small_mask, a, b)
    if which.startswith("上"):                      # user asked for the other arc
        cnts, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
        ia = int(np.argmin(np.hypot(cnt[:, 0] - a[0], cnt[:, 1] - a[1])))
        ib = int(np.argmin(np.hypot(cnt[:, 0] - b[0], cnt[:, 1] - b[1])))
        if ia > ib:
            ia, ib = ib, ia
        arc1, arc2 = cnt[ia:ib + 1], np.vstack([cnt[ib:], cnt[:ia + 1]])
        arc = arc1 if arc1[:, 1].mean() < arc2[:, 1].mean() else arc2

    # smooth at working scale, back to full-res, then translate straight up
    line = edge.smooth_axis(arc, tol=float(smooth_tol)) / f
    line = edge.offset_up(line, float(offset_pct))

    base = BGR if show_on.startswith("抠") else imaging.load_bgr(str(config.DEFAULT_IMAGE))
    vis = base.copy()
    cv2.polylines(vis, [np.round(line).astype(np.int32)], False, (0, 255, 255), 6, cv2.LINE_AA)
    cv2.circle(vis, (int(pts[0][0]), int(pts[0][1])), 20, (0, 0, 255), -1)
    cv2.circle(vis, (int(pts[1][0]), int(pts[1][1])), 20, (0, 255, 0), -1)

    ts = time.strftime("%H%M%S")
    png = config.OUTPUT_DIR / f"edge_{ts}.png"
    cv2.imwrite(str(png), vis)
    np.savetxt(config.OUTPUT_DIR / f"edge_{ts}.csv", line, delimiter=",", header="x,y", comments="")
    length = float(np.sum(np.hypot(*np.diff(line, axis=0).T)))
    info = (f"**{which}**,上移 {offset_pct:.0f}px | P1=({pts[0][0]:.0f},{pts[0][1]:.0f}) "
            f"P2=({pts[1][0]:.0f},{pts[1][1]:.0f}) | 线长 {length:.0f}px | 已存 outputs/{png.name}")
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), info


def on_click(pts, side, which, offset_pct, smooth_tol, show_on, evt: gr.SelectData):
    x, y = evt.index
    pts = list(pts or [])
    if len(pts) >= 2:
        pts = []
    pts.append((float(x), float(y)))
    marked = _marks(RGB, pts)
    if len(pts) == 1:
        return marked, pts, None, "已点 **P1**。再点 **P2**(另一端)。"
    img, info = _compute(pts, side, which, offset_pct, smooth_tol, show_on)
    return marked, pts, img, info


def on_param(pts, side, which, offset_pct, smooth_tol, show_on):
    if not pts or len(pts) < 2:
        return None, "先点两个端点。"
    return _compute(pts, side, which, offset_pct, smooth_tol, show_on)


def on_reset():
    return RGB, [], None, "已重置,请点 **P1**。"


with gr.Blocks(title="Shell edge line") as demo:
    gr.Markdown(
        "## 壳缘线工具 · 第一步:两点之间的边缘线\n"
        "底图是 **SAM 抠出的干净壳**(已去掉碎屑/比例尺/背景)。在左图点**两个端点**,右图给出它们之间的边缘线。"
        "第三次点击重新开始。**上移量**可把线从壳缘往壳内推(按局部壳厚的百分比)。"
    )
    pts_state = gr.State([])
    with gr.Row():
        which = gr.Radio(["下缘(腹缘)", "上缘(背缘)"], value="下缘(腹缘)", label="取哪一侧")
        show_on = gr.Radio(["抠出的壳", "原图"], value="抠出的壳", label="画在哪张图上")
        side = gr.Slider(600, 4200, value=1600, step=100, label="计算分辨率")
    with gr.Row():
        offset = gr.Slider(0, 600, value=0, step=10, label="上移量(像素,整条线垂直上移)")
        smooth = gr.Slider(0.5, 10, value=2.5, step=0.5, label="平滑度")
    with gr.Row():
        inp = gr.Image(value=RGB, type="numpy", label="点两个端点", interactive=True)
        out = gr.Image(type="numpy", label="边缘线结果")
    status = gr.Markdown("请点 **P1**,再点 **P2**。")
    reset = gr.Button("重置")

    ctrl = [pts_state, side, which, offset, smooth, show_on]
    inp.select(on_click, ctrl, [inp, pts_state, out, status])
    reset.click(on_reset, None, [inp, pts_state, out, status])
    for k in (which, show_on, side, offset, smooth):
        k.change(on_param, ctrl, [out, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, show_error=True)
