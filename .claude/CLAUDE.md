# A4/A5 Object Detection — Session Brief

> **Paste this at the start of a session.** It tells Claude how to work with me and
> what's already decided. Update the Progress table before closing a session.

---

## How to work with me

I'm working through the EECS 498/598 detection assignments (FCOS + Faster R-CNN) as a
self-studier — no grades, no submission. I want to spend effort only on what reflects
real CV work; mechanical tensor plumbing that the docstring already dictates is work
I'd delegate in a real job.

**Rules:**
- **Be concise.** Short answers, small window, I have ADHD. No long preambles.
- I work **one function at a time**. I'll select it in the IDE or name it.
- **🟢 = you write it.** Just do it, no walkthrough.
- **🔴 = you explain, I write.** Review my attempt, point at bugs, don't hand me code
  unless I ask.
- Don't use plan mode for this. Sanity cells catch mistakes in seconds.

---

## Progress

| Block | File | Status |
|---|---|---|
| `DetectorBackboneWithFPN.__init__` | common.py | ✅ done (Claude) |
| `DetectorBackboneWithFPN.forward` | common.py | ✅ done (me) — uses `bilinear`, paper uses `nearest`; either fine |
| `get_fpn_location_coords` | common.py | ✅ done (me) |
| `nms` | common.py | ✅ pasted from Appendix |
| `FCOSPredictionNetwork.__init__` / `forward` | one_stage | ✅ done (Claude) |
| `fcos_get_deltas_from_locations` | one_stage | ✅ done (me, Claude vectorized) |
| `fcos_apply_deltas_to_locations` | one_stage | ✅ done (me, Claude fixed signs/clamp) |
| `fcos_make_centerness_targets` | one_stage | ✅ done (Claude) |
| `FCOS.__init__` + `forward` wiring/matching | one_stage | ✅ done (Claude) |
| `FCOS.forward` losses | one_stage | ✅ done (me) — `0.25 *` on box loss left out deliberately, see below |
| `FCOS.inference` | one_stage | ✅ done (Claude) |
| everything in `two_stage_detector.py` | | ⬜ not started |

**Nothing has run yet.** All of the above is syntax-checked only; torch isn't installed locally.

### Open items
- Box loss weight: notebook cell 43 uses `0.25 * F.l1_loss(...)`; I'm running without
  it first as a baseline. If mAP is low, that's the first knob.
- Colab verification order: nb1 #33 → #38 → #40 → #49 → loss cell (~#43) → overfit (#45)
  → full run → `cd mAP && python main.py`.

---

## The triage

### 🟢 Claude writes these — fully dictated by docstring

| Where | Why it's mechanical |
|---|---|
| `FCOSPredictionNetwork` stem + heads | kernel/stride/padding/init all given |
| `RPNPredictionNetwork` stem + heads | file says "okay to use your FCOS implementation" |
| `FasterRCNN.__init__` stem + flatten | third copy of the same stem pattern |
| `FCOS.__init__`, `FCOS.forward` backbone pass | 2-line wiring |
| `generate_fpn_anchors` | docstring gives the literal formulas |
| `rcnn_get_deltas_from_anchors` | formula on lecture Slide 68, linked in file |
| `roi_align` call | "call `torchvision.ops.roi_align`, use `aligned=True`" |
| `FasterRCNN` classifier loss (~839) | sampling ratio, loss fn, label-shift all handed over |
| `FCOS.inference` (~564) | 4 numbered steps; sqrt(cls·ctr) line pre-written |
| `RPN.predict_proposals` (~637) | 3 steps + `torch.topk` hint + shape warning |
| `FasterRCNN.inference` (~913) | **no TODO token** — banner only |
| `nms` | 3 numbered steps + link to torchvision C++ source |

### 🔴 I write these — the actual job

**1. `iou` (`two_stage_detector.py:210`)** — one sentence, zero hints. M×N broadcasting,
clamp negative overlap. Gates `rcnn_match_anchors_to_gt` → 3 downstream blocks.

**2. Label assignment — the most important idea in detection.**
Read the *provided* `fcos_match_locations_to_gt` first, then write `RPN.forward`
matching + loss (~535, ~571). Anchor-free vs anchor-based, IoU thresholds,
fg/bg/neutral, positive sampling. Usual cause of "trains fine, predicts nothing."

**3. `FCOS.forward` losses (~487)** — largest block, 15-25 lines. Only background-zeroing
is stated. Which loss per head, one-hot encoding, +1 class shift: all mine. The
bookkeeping (masking, EMA normalizer, normalize by foreground count) is the classic
silent detection bug.

**4. `RPNPredictionNetwork.forward` reshape (~105)** — looks mechanical, isn't. Anchor dim
must interleave as `(H, W, A)` to match `generate_fpn_anchors`. Nothing warns about this.

**5. Everything after training** ← *most students skip this; it's the actual day job.*
- Overfit-small-data first (cell 45 / 38) — standard first move in any real training run
- Reading loss curves, deciding if it's working
- Thresholds: notebook uses **0.5/0.5 to visualize, 0.4/0.6 for mAP**. Why they differ
  is real precision/recall reasoning.
- Per-class AP from `mAP/` — explain *why* specific classes fail

### ⚪ Just run
Colab/Drive setup, dataset download, pre-filled hyperparameters
(`lr=8e-3`, `max_iters=9000`, `fpn_channels=128`). Skip the submission zip — not
submitting anything.

---

## Traps

- **3 blocks have no `TODO:` token** — `two_stage_detector.py` ~546, ~829, ~928.
  Grep on `TODO` undercounts by three. Search `END OF YOUR CODE` instead.
- **`fcos_apply_deltas_to_locations`**: `output_boxes` returned but never initialized.
- **`iou`**: local variable shadows the function name.
- **RPN sentinels differ from FCOS**: deltas `-1e8` (not `-1`), objectness target `0`
  (not `-1`).
- **Dependency chain**: `common.nms` gates both detectors' inference. `two_stage.iou`
  gates matching → RPN losses → second-stage matching.
- **torch is not installed locally** (Python 3.13). Nothing runs on this machine —
  everything is verified in Colab. Also why VS Code gives no `nn.` autocomplete.

---

## Mental models I've already worked out — don't re-explain

- A feature map is a **dense grid**, not a set of interesting points. Every cell has a
  feature. "Feature" here ≠ SIFT/Harris feature.
- Conv **computes**, it doesn't **check**. The head decides what's interesting later.
- Weights are shared across all cells and fixed after training; only the *output* varies
  per location.
- FPN is a **neck**, not a backbone: `backbone → FPN → head`. Both detectors use it.
- Faster R-CNN = FCOS pipeline + propose-then-refine (RPN → RoIAlign → head).
  RoIAlign is *not* replaced by FPN; FPN just gives it three levels to crop from.
- `F.interpolate` is plain resizing, no learned weights (unlike `ConvTranspose2d`).

---

## Verification

Every block has a notebook sanity cell with hard-coded expected values via `rel_error`:

| Cell | Checks |
|---|---|
| nb1 #33 | location coords vs `expected_locations` |
| nb1 #38 | delta round-trip + background = −1 |
| nb1 #40 | centerness vs `[0.3086…, 0.1667, −1.0]`, rel err < 1e-7 |
| nb1 #49 | NMS vs `torchvision.ops.nms`, diff < 1% (mine will be slower — expected) |
| nb2 #30 | `iou` vs `expected_iou` |
| nb2 #34 | R-CNN delta round-trip |

Then: overfit cell → loss near zero → full 9000-iter run → `cd mAP && python main.py`.

**Course bar: ≥22% mAP for FCOS** — a useful self-check, not a grade. No bar given for
Faster R-CNN; the notebook says it's expected to underperform (smaller model, short
training, second-stage box regression deliberately omitted).

---

## Appendix: `nms` — ready to paste

Replaces the `pass` at ~line 200. Delete the `keep = None` line above the banner.

```python
    keep = []
    order = scores.argsort(descending=True)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]

        inter_x1 = torch.maximum(x1[i], x1[rest])
        inter_y1 = torch.maximum(y1[i], y1[rest])
        inter_x2 = torch.minimum(x2[i], x2[rest])
        inter_y2 = torch.minimum(y2[i], y2[rest])

        # clamp: non-overlapping boxes give negative width/height
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        ious = inter / (areas[i] + areas[rest] - inter)

        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
```
