"""Shell-to-line studio: click SAM prompts, pick a mask, trace and save a line.

Four steps, all in one page:

  1. Click positive points on the shell and negative points on anything to
     exclude (debris, scale bar, background). Run SAM.
  2. SAM returns several candidate masks -- pick the one that is actually the
     shell. (SAM emits an ambiguity hierarchy; choosing the granularity is a
     judgement call it cannot make for us.)
  3. On the extracted shell, click the two endpoints of the margin, choose which
     side, and shift the line up by however many pixels.
  4. Save. The line CSV is the handoff to the ring-detection step, which samples
     brightness along it and finds the dark-band valleys.

Must run in the GPU env, since SAM needs torch:

    D:\\Anaconda\\envs\\sea\\python.exe studio.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

import config
from surfclam import imaging, edge

RESULTS = config.PROJECT_DIR / "results"
RESULTS.mkdir(exist_ok=True)
SAM_CKPT = config.PROJECT_DIR / "models" / "sam_vit_h_4b8939.pth"

POS_COLOR = (0, 200, 0)      # RGB
NEG_COLOR = (230, 30, 30)


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------
def list_images() -> dict[str, str]:
    d = config.DATA_DIR / "Surf Clam hinge images"
    out: dict[str, str] = {}
    default = config.DEFAULT_IMAGE
    if default.exists():
        out[f"[默认] {default.name}"] = str(default)
    for sub in ["NOAA Surf Clam Survey/Reader comparison selected images",
                "NOAA Surf Clam Survey/1986", "NOAA Surf Clam Survey/2015",
                "NOAA Surf Clam Survey/2016", "NOAA Surf Clam Survey/2011"]:
        p = d / sub
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.jpg"))[:12]:
            out.setdefault(f"{Path(sub).name}/{f.name}", str(f))
    return out or {"(none)": str(config.DEFAULT_IMAGE)}


IMAGES = list_images()


def load_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(imaging.load_bgr(path), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# SAM, loaded once and kept warm; the image embedding is cached per image
# ---------------------------------------------------------------------------
_SAM: dict = {"predictor": None, "key": None}


def get_predictor():
    if _SAM["predictor"] is None:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        if not SAM_CKPT.exists():
            raise gr.Error(f"缺少 SAM 权重: {SAM_CKPT}")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT)).to(device=dev)
        _SAM["predictor"] = SamPredictor(sam)
        _SAM["device"] = dev
    return _SAM["predictor"]


def sam_masks(img_path: str, pts: list) -> tuple[list[np.ndarray], list[float]]:
    """pts: [(x, y, label)]  label 1 = positive, 0 = negative."""
    pred = get_predictor()
    if _SAM["key"] != img_path:          # embedding is the expensive part
        pred.set_image(load_rgb(img_path))
        _SAM["key"] = img_path
    coords = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
    labels = np.array([p[2] for p in pts], dtype=np.int32)
    masks, scores, _ = pred.predict(point_coords=coords, point_labels=labels,
                                    multimask_output=True)
    return [m.astype(np.uint8) for m in masks], [float(s) for s in scores]


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_prompts(rgb: np.ndarray, pts: list) -> np.ndarray:
    out = rgb.copy()
    r = max(10, out.shape[1] // 150)
    for x, y, lb in pts:
        col = POS_COLOR if lb == 1 else NEG_COLOR
        cv2.circle(out, (int(x), int(y)), r, col, -1)
        cv2.circle(out, (int(x), int(y)), r, (255, 255, 255), 3)
        sym = "+" if lb == 1 else "-"
        cv2.putText(out, sym, (int(x) - r // 2, int(y) + r // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, r / 16, (255, 255, 255), max(2, r // 6))
    return out


def draw_mask(rgb: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    out = rgb.copy()
    tint = np.zeros_like(out)
    tint[mask > 0] = (255, 60, 60)
    out = cv2.addWeighted(out, 1.0, tint, 0.40, 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 4)
    cv2.putText(out, label, (26, 76), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 5)
    return out


def cut_shell(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask == 0] = 0
    return out


def draw_line(base_rgb, line, p1=None, p2=None):
    out = base_rgb.copy()
    cv2.polylines(out, [np.round(line).astype(np.int32)], False, (255, 230, 0), 6, cv2.LINE_AA)
    for p, col in ((p1, (255, 0, 0)), (p2, (0, 255, 0))):
        if p is not None:
            cv2.circle(out, (int(p[0]), int(p[1])), 20, col, -1)
    return out


# ---------------------------------------------------------------------------
# step 1: prompts -> masks
# ---------------------------------------------------------------------------
def on_pick_image(name):
    rgb = load_rgb(IMAGES[name])
    return rgb, [], None, [], None, None, [], "已载入图片。用**正点**点在壳上,**负点**点在要排除的东西上。"


def on_prompt_click(name, pts, mode, evt: gr.SelectData):
    x, y = evt.index
    pts = list(pts or []) + [(float(x), float(y), 1 if mode.startswith("正") else 0)]
    n_pos = sum(1 for p in pts if p[2] == 1)
    n_neg = len(pts) - n_pos
    return draw_prompts(load_rgb(IMAGES[name]), pts), pts, f"正点 {n_pos} 个,负点 {n_neg} 个。"


def on_undo(name, pts):
    pts = list(pts or [])[:-1]
    return draw_prompts(load_rgb(IMAGES[name]), pts), pts, f"已撤销,剩 {len(pts)} 个点。"


def on_clear(name):
    return load_rgb(IMAGES[name]), [], "已清空所有点。"


def on_run_sam(name, pts):
    if not pts or not any(p[2] == 1 for p in pts):
        raise gr.Error("至少需要一个正点(点在壳上)。")
    rgb = load_rgb(IMAGES[name])
    masks, scores = sam_masks(IMAGES[name], pts)
    h, w = masks[0].shape
    gallery, choices = [], []
    for i, (m, s) in enumerate(zip(masks, scores)):
        area = 100.0 * m.sum() / (h * w)
        pieces = cv2.connectedComponents(m, 8)[0] - 1
        lab = f"#{i}  score={s:.3f}  area={area:.1f}%  pieces={pieces}"
        gallery.append(draw_mask(rgb, m, lab))
        choices.append(f"#{i} (score {s:.3f}, area {area:.1f}%)")
    return (gallery, masks, gr.update(choices=choices, value=choices[0]),
            f"SAM 给出 {len(masks)} 个候选,选一个继续。")


# ---------------------------------------------------------------------------
# step 2/3: choose mask -> click endpoints -> line
# ---------------------------------------------------------------------------
def on_choose_mask(name, masks, choice):
    if not masks:
        raise gr.Error("请先运行 SAM。")
    idx = int(choice.split()[0].lstrip("#"))
    mask = masks[idx]
    cut = cut_shell(load_rgb(IMAGES[name]), mask)
    return cut, mask, [], None, f"已选 mask #{idx}。在下图点**两个端点**。"


def compute_line(mask, pts, side, dy, tol, work):
    h, w = mask.shape
    f = min(1.0, work / max(h, w))
    small = cv2.resize(mask, (int(w * f), int(h * f)), interpolation=cv2.INTER_NEAREST) * 255
    a = (pts[0][0] * f, pts[0][1] * f)
    b = (pts[1][0] * f, pts[1][1] * f)

    cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise gr.Error("掩码为空。")
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    ia = int(np.argmin(np.hypot(cnt[:, 0] - a[0], cnt[:, 1] - a[1])))
    ib = int(np.argmin(np.hypot(cnt[:, 0] - b[0], cnt[:, 1] - b[1])))
    if ia > ib:
        ia, ib = ib, ia
    arc1, arc2 = cnt[ia:ib + 1], np.vstack([cnt[ib:], cnt[:ia + 1]])
    want_lower = side.startswith("下")
    arc = (arc1 if (arc1[:, 1].mean() > arc2[:, 1].mean()) == want_lower else arc2)
    if np.hypot(*(arc[0] - np.array(a))) > np.hypot(*(arc[-1] - np.array(a))):
        arc = arc[::-1]

    line = edge.smooth_axis(arc, tol=float(tol)) / f
    return edge.offset_up(line, float(dy))


def on_line_click(name, mask, pts, side, dy, tol, work, evt: gr.SelectData):
    if mask is None:
        raise gr.Error("请先选一个 mask。")
    x, y = evt.index
    pts = list(pts or [])
    if len(pts) >= 2:
        pts = []
    pts.append((float(x), float(y)))
    cut = cut_shell(load_rgb(IMAGES[name]), mask)
    if len(pts) == 1:
        out = cut.copy()
        cv2.circle(out, (int(x), int(y)), 20, (255, 0, 0), -1)
        return out, pts, None, "已点 P1,再点 P2。"
    line = compute_line(mask, pts, side, dy, tol, work)
    return (draw_line(cut, line, pts[0], pts[1]), pts, line,
            f"线长 {np.sum(np.hypot(*np.diff(line, axis=0).T)):.0f}px,可拖动**上移量**微调。")


def on_line_param(name, mask, pts, side, dy, tol, work):
    if mask is None or not pts or len(pts) < 2:
        return None, None, "先选 mask 并点两个端点。"
    line = compute_line(mask, pts, side, dy, tol, work)
    cut = cut_shell(load_rgb(IMAGES[name]), mask)
    return (draw_line(cut, line, pts[0], pts[1]), line,
            f"上移 {dy:.0f}px,线长 {np.sum(np.hypot(*np.diff(line, axis=0).T)):.0f}px。")


# ---------------------------------------------------------------------------
# step 4: save
# ---------------------------------------------------------------------------
def on_save(name, mask, line, pts, sam_pts, side, dy, tol, tag):
    if line is None or mask is None:
        raise gr.Error("还没有线可保存。")
    stem = Path(IMAGES[name]).stem
    tag = (tag or "").strip().replace(" ", "_")
    folder = RESULTS / f"{stem}{'_' + tag if tag else ''}_{time.strftime('%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb(IMAGES[name])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    line = np.asarray(line)

    np.savetxt(folder / "line.csv", line, delimiter=",", header="x,y", comments="", fmt="%.2f")
    cv2.imwrite(str(folder / "mask.png"), (mask > 0).astype(np.uint8) * 255)
    np.save(folder / "mask.npy", (mask > 0).astype(np.uint8))
    cv2.imwrite(str(folder / "overlay.png"),
                cv2.cvtColor(draw_line(cut_shell(rgb, mask), line, pts[0], pts[1]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder / "overlay_on_photo.png"),
                cv2.cvtColor(draw_line(rgb, line, pts[0], pts[1]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder / "shell_cut.png"), cv2.cvtColor(cut_shell(rgb, mask), cv2.COLOR_RGB2BGR))

    meta = {
        "image": IMAGES[name],
        "sam_points": [{"x": p[0], "y": p[1], "label": int(p[2])} for p in (sam_pts or [])],
        "endpoints": [list(pts[0]), list(pts[1])],
        "side": side,
        "offset_up_px": float(dy),
        "smooth_tol": float(tol),
        "line_points": int(len(line)),
        "line_length_px": float(np.sum(np.hypot(*np.diff(line, axis=0).T))),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ 已保存到 `results/{folder.name}/`(line.csv, mask.png/npy, overlay*.png, shell_cut.png, meta.json)"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Surf Clam studio") as demo:
    gr.Markdown(
        "# 海蚶壳 · 取线工作台\n"
        "**① 点正负点 → 跑 SAM ② 选 mask ③ 点两端点取线、上移 ④ 保存**  \n"
        "保存的 `line.csv` 就是后续「沿线取灰度、找暗带波谷数生长环」那一步的输入。"
    )
    img_name = gr.Dropdown(choices=list(IMAGES), value=list(IMAGES)[0], label="图片")

    sam_pts = gr.State([])
    masks_st = gr.State([])
    mask_st = gr.State(None)
    line_pts = gr.State([])
    line_st = gr.State(None)

    with gr.Tab("① 点正负点 → SAM"):
        with gr.Row():
            mode = gr.Radio(["正点 + (壳上)", "负点 - (要排除的)"], value="正点 + (壳上)", label="点击类型")
            btn_undo = gr.Button("撤销上一点")
            btn_clear = gr.Button("清空")
            btn_run = gr.Button("运行 SAM", variant="primary")
        prompt_img = gr.Image(value=load_rgb(list(IMAGES.values())[0]), type="numpy",
                              label="在图上点(正点=绿,负点=红)", interactive=True)
        s1 = gr.Markdown("用**正点**点在壳上,**负点**点在碎屑/比例尺/背景上,然后运行 SAM。")

    with gr.Tab("② 选 mask"):
        gallery = gr.Gallery(label="SAM 候选", columns=1, height=460)
        pick = gr.Radio(choices=[], label="选哪一个")
        btn_pick = gr.Button("用这个 mask", variant="primary")
        s2 = gr.Markdown("先在第①步运行 SAM。")

    with gr.Tab("③ 取线 + 上移"):
        with gr.Row():
            side = gr.Radio(["下缘(腹缘)", "上缘(背缘)"], value="下缘(腹缘)", label="取哪一侧")
            dy = gr.Slider(0, 600, value=0, step=10, label="上移量(像素)")
        with gr.Row():
            tol = gr.Slider(0.5, 10, value=2.5, step=0.5, label="平滑度")
            work = gr.Slider(600, 4200, value=1600, step=100, label="计算分辨率")
        line_img = gr.Image(type="numpy", label="点两个端点", interactive=True)
        s3 = gr.Markdown("先在第②步选一个 mask。")

    with gr.Tab("④ 保存"):
        tag = gr.Textbox(label="备注(可选,加进文件夹名)", placeholder="line1")
        btn_save = gr.Button("保存线和图", variant="primary")
        s4 = gr.Markdown("保存 line.csv / mask / overlay / meta.json 到 `results/`。")

    # step 1
    img_name.change(on_pick_image, [img_name],
                    [prompt_img, sam_pts, gallery, masks_st, mask_st, line_img, line_pts, s1])
    prompt_img.select(on_prompt_click, [img_name, sam_pts, mode], [prompt_img, sam_pts, s1])
    btn_undo.click(on_undo, [img_name, sam_pts], [prompt_img, sam_pts, s1])
    btn_clear.click(on_clear, [img_name], [prompt_img, sam_pts, s1])
    btn_run.click(on_run_sam, [img_name, sam_pts], [gallery, masks_st, pick, s2])
    # step 2
    btn_pick.click(on_choose_mask, [img_name, masks_st, pick], [line_img, mask_st, line_pts, line_st, s3])
    # step 3
    line_img.select(on_line_click, [img_name, mask_st, line_pts, side, dy, tol, work],
                    [line_img, line_pts, line_st, s3])
    for k in (side, dy, tol, work):
        k.change(on_line_param, [img_name, mask_st, line_pts, side, dy, tol, work],
                 [line_img, line_st, s3])
    # step 4
    btn_save.click(on_save, [img_name, mask_st, line_st, line_pts, sam_pts, side, dy, tol, tag], [s4])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False, show_error=True)
