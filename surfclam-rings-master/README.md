# Surf Clam Growth Rings

Tools for finding growth rings in photographs of surf clam hinge sections.

The work splits into two independent halves:

1. **Draw a line** across the shell, from one end to the other.
2. **Detect rings** by reading the brightness along that line: a growth band is
   dark, so it shows up as a dip.

Both halves have several implementations. `detect.py` runs any combination.

---

## Quick start

```
D:\Anaconda\envs\sea\python.exe detect.py --image "path\to\photo.jpg"
```

That uses the current best defaults: an automatic line, and peaks/valleys read
straight off the pixels. It writes into `outputs/detect_auto_raw/`:

| file | what it is |
| --- | --- |
| `<name>.png` | the photo with the line and the detected bands drawn on it |
| `<name>_rings.csv` | one row per detected band: `x, y, strength` |
| `<name>_line.csv` | the line itself, as x/y pixel coordinates |
| `<name>_meta.json` | which methods and settings produced this |

To see every method available:

```
python detect.py --list
```

---

## Part 1 — Drawing the line

### `auto` — automatic, no clicking *(default)*

A small U-Net predicts a thin band down the middle of the shell, and the line is
read out of it. **This model was trained in an earlier project** (`D:\Surf-Clam-`)
and copied here — we did not train it. The weights are `models/shell_unet.pt`;
the code is `surfclam/autoline.py`, rewritten as one self-contained file and
checked to reproduce the original output pixel for pixel.

It runs in about half a second per image on the GPU. Across the 38
reader-comparison images it produced a line on all 38, and on a median 91% of
each line the model reports a real prediction rather than a gap it bridged and
flagged.

**Its weakness:** the line is a *shape* centreline, not a measurement axis. On
some shells it drifts onto the dark shadow at the lower margin, where there are
no rings to find, and detection there returns mostly edge noise. When results
look poor, check where the line went before blaming the ring detector.

### `edge` — trace the shell margin, then move inward

You give two endpoints. The shell outline between them is traced along either
the lower (ventral) or upper (dorsal) side, then shifted inward by a fixed
number of pixels. Simple and predictable, and you control where the line sits.

```
python detect.py --image photo.jpg --line edge --p1 1150,600 --p2 4100,400 --shift 120
```

Shift is in pixels: positive moves up, negative moves down. Use negative when
tracing the upper edge, since inward is downward there.

### `center` — medial axis

The geometric centre of the shell mask between two points. Kept for comparison.
It follows the *outline* of the shell, so it wanders wherever one side is
thicker, and it knows nothing about the rings.

### `saved` — reuse a line you drew by hand

```
python detect.py --line saved --line-csv results\<folder>\line.csv
```

Lines drawn in `pick_line.py` are saved in that format.

### Drawing lines by hand: `pick_line.py`

An OpenCV window that walks through a folder of images. For each one:

1. Left-click positive points on the shell, right-click negative points on
   anything to exclude (debris, the scale bar, background). Press ENTER to run
   SAM and SAM 2.
2. Page through the candidate masks with `n`/`p` and accept one with ENTER, or
   press `b` to add more points and try again.
3. Click the two endpoints, adjust the shift with `+`/`-`, switch between the
   upper and lower edge with `t`.
4. Press `s` to save and move to the next image.

```
D:\Anaconda\envs\sea\python.exe pick_line.py --skip-done
```

All on-screen text is ASCII, because OpenCV can only draw Hershey fonts and
renders anything else as `?`.

---

## Part 2 — Detecting rings

All methods read the grey values along the line and look for dark dips. They
differ in what they do to that profile first.

### `raw` — peaks and valleys off the pixel values *(default, best)*

CLAHE for contrast, a light smooth, then `find_peaks`. Valleys are the dark
growth bands; peaks are the bright increments between them. That is all.

This is the original, simplest approach, and on inspection it beats everything
more elaborate that we tried. Two of those elaborations turned out to be
actively harmful, and are worth naming so nobody re-adds them:

- **Averaging the profile along the ring direction** was supposed to raise the
  signal. Measured band contrast: **11.35** grey levels with no averaging,
  **10.59** at the best averaged setting, worse at every wider setting. The ring
  direction cannot be estimated accurately enough to integrate tens of pixels
  along it without smearing the band.
- **Local contrast normalisation** divides by a running standard deviation. That
  equalises contrast along the line — and so *suppresses* a faint band that sits
  in a stretch already busy with bands. Exactly backwards when the goal is to
  miss nothing.

### `global`

The same as `raw` but on the contrast-normalised profile. Kept for comparison.

### `warped`

Ring spacing changes several-fold from one end of the shell to the other, so no
single minimum distance fits the whole line. This warps the axis so the spacing
becomes roughly uniform, detects there, and warps back.

### `dp`

Chooses the whole sequence of rings at once with dynamic programming, rewarding
dark bands and penalising spacings that disagree with the local trend. In
principle the most principled; in practice it under-counts badly.

### `multi`

Runs several parallel lines (±15 and ±30 px) and pools them **by union**, so a
band that is faint where one line crosses it can still be caught by another. How
many lines saw a band becomes a confidence score (`votes`), not a filter.
Detections are matched across lines by projecting along the band's own
direction, because the bands are tilted and the same band crosses lines 30 px
apart at quite different x.

---

## Turning the sensitivity up

The point is **not** to output the correct number of rings. It is to surface
every candidate band, including the sub-annual "false" checks, and let a person
decide which is which — those checks record real events and matter in their own
right.

Two knobs, both "smaller = more detections":

```
python detect.py --image photo.jpg --prominence 0.008 --min-dist 3
```

On one image, `--prominence` alone moves the count like this:

| `--prominence` | rings found |
| --- | --- |
| 0.08 | 23 |
| 0.03 | 33 |
| 0.015 *(default)* | 50 |
| 0.008 | 60 |

Every detection carries a `strength` (0–1) in the CSV, so you can sort and
review the confident ones first instead of treating them all alike.

---

## What is still unresolved

Stated plainly, because it affects how far you should trust the output:

- **We detect dark bands, not annual rings.** Nothing in the pipeline separates
  a true annulus from a sub-annual check, a shell lamella, a scratch, or the
  shadow at the shell margin. That separation still needs a human.
- **Line placement now limits quality more than the detector does.** The same
  detector settings give good results where the line runs through clear banding
  and poor results where it runs along the margin.
- **Spacing in physical units is not implemented.** Every distance is in pixels.
  Each photo carries a 2 mm scale bar that would give the conversion, and the
  dataset has calibration spreadsheets, but neither is wired in.
- **Reader ages are only a partial check.** `Results2.xlsx` lists an age per
  image, but its increment columns stop at `I13`, so for the 21 of 38 shells
  older than 13 the later increments were never recorded.

---

## Layout

```
detect.py            run any line method with any ring method
pick_line.py         draw lines by hand (SAM / SAM 2 + OpenCV window)
studio.py            same workflow in a browser (Gradio)
config.py            paths and shared parameters

surfclam/
  autoline.py        U-Net centreline, ported from the earlier project
  ringdet.py         all ring-detection methods
  edge.py            trace the shell margin between two points
  centerline.py      medial axis
  refine.py          mask cleanup: SAM refinement, snap boundary to real edges
  locate.py          find the shell in the photo
  imaging.py         loading and greyscale
  extract.py         earlier whole-ring extraction experiments
  growth_axis.py     ring orientation via the structure tensor

models/              weights (gitignored, not in version control)
outputs/             everything written by the scripts
results/             lines saved from pick_line.py / studio.py
data/                the photographs
```

## Environment

The GPU environment has torch, SAM, SAM 2 and the U-Net:

```
D:\Anaconda\envs\sea\python.exe
```

Do not `pip install` a package that lists torch as a dependency into it without
pinning: installing `sam2` once replaced the CUDA build of torch with a CPU-only
one and silently disabled the GPU. Restore with:

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```
