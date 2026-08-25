# ComfyUI Adaptive Edge Fade

A standalone ComfyUI custom node for softening foregrounds that were cut by the canvas edge, without treating every harmless edge touch as overflow.

The node uses a precision-first crop-evidence gate, emits alpha-safe `1280 × 1280` and `168 × 168` RGBA images, and requires no model, network request, telemetry, or external asset.

## What changed in v0.2.7

v0.2.7 changes the final fade unit from an individual run/component to the canvas side:

- Crop evidence is still evaluated conservatively per candidate segment.
- One confirmed crop activates exactly one shared circular field for that side (`top`, `right`, `bottom`, or `left`).
- Every visible alpha pixel inside that active side's near-edge transition band follows the same curve, even when it belongs to another disconnected component.
- A skipped tangent or harmless edge touch cannot activate a side by itself. If another segment does activate that side, the skipped object participates only in the shared soft transition.
- Only the real confirmed source runs receive the exact transparent safety inset. The node never clears an entire edge.
- Multiple active sides are combined with normalized smooth edge weights, not multiplication or a hard `min`, avoiding dark square corners and diagonal seams.
- Opposite active sides complete before the centerline so they do not darken the middle of the icon.
- The production default for `feather_width` is now `160`; `240` remains available when a deeper transition is desired.

This side-level behavior is intentional. It prevents the visible split produced when two clipped legs, handles, or other runs were faded independently while the body or a nearby object remained untouched.

## Detection behavior

- The narrow outer probe only discovers candidates; edge contact alone does not trigger a fade.
- A candidate must contain a sufficiently long and dense actual-border core, stable inward support, and persistent width before it is confirmed as a crop cut.
- Unconfirmed fragments remain independent during detection. Multiple weak fragments and opposite-side touches cannot promote one another into a crop.
- Natural tangencies, anti-aliased endings, shallow edge slivers, sparse residue, and symmetric frame-filling components are preserved when evidence is ambiguous.
- Nearly uniform low-alpha restore residue spanning most of an edge is classified as a weak component halo instead of a crop.
- Narrow chords that widen on both sides remain conservatively skipped because a complete round shoulder and a cropped narrow neck can be pixel-identical after cropping.
- Confirmed source segments keep their exact safety cores; gaps and other components receive only the shared smooth side field.
- The side field uses a true-circle signed-distance curve and fifth-order smootherstep. It returns to exact guard `1` after the completion depth and cannot reappear on the far side of a long non-square image.
- Top, right, bottom, and left are rotation/mirror equivalent.

## Alpha and resizing

- RGB and alpha are resized together in linear-light premultiplied form.
- Non-square inputs use transparent contain-padding; edge pixels are never replicated.
- The `168 × 168` icon is derived from the repaired 1280 master.
- Only very-low-alpha straight-RGB residue is attenuated. Nonzero alpha is not globally clipped or remapped.
- If `rgba` already contains alpha, it is intersected with `final_alpha`; pixels hidden by either source cannot be revived.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aoiro0X0/comfyui-adaptive-edge-fade.git
```

Restart ComfyUI, then search for:

```text
Adaptive Edge Fade + Alpha-safe 1280/168
```

## Inputs

- `rgba`: ComfyUI `IMAGE`, RGB or RGBA.
- `final_alpha`: visible-alpha `MASK`; `1 = visible`, `0 = transparent`. Do not invert it.
- `safety_inset`: exact transparent inset on confirmed source segments, expressed in the canonical 1280 output space. Default: `8`.
- `feather_width`: inward circular transition width in the canonical 1280 output space. Default and recommended automatic-pipeline value: `160`.
- `contact_threshold`: threshold used to estimate the deeper body span. Default: `0.03`. The outer candidate probe intentionally uses a much lower threshold.
- `rgb_cleanup_alpha`: very-low-alpha range in which residual straight RGB is attenuated. Default: `0.06`.

Existing workflows load without rewiring because the node class, six inputs, five outputs, names, types, and order are unchanged. ComfyUI preserves saved widget values, so a workflow saved with v0.2.6 may still show `feather_width=240`; set it to `160` or import the updated workflow if you want the new production preset.

## Outputs

- `icon_1280_rgba`
- `icon_168_rgba`
- `final_alpha_1280`
- `edge_guard_1280`
- `diagnostics`

Diagnostics use `clip_evidence_v9`. They include four `side_activations` records and mark processed contacts with:

- `mode=side_circle_arc`
- `side_component_scope=all_visible_near_edge`
- `side_field_combination=normalized_active_edge_partition`

## Limits

The node can soften geometry already cut by the canvas boundary; it cannot reconstruct missing off-canvas pixels. A complete flat edge that naturally ends on the last row can be pixel-identical to the same shape continuing outside the canvas. v0.2.7 deliberately prefers a false negative when alpha-only evidence is ambiguous. Exact overflow classification requires upstream overscan or a per-segment overflow seed mask.

## Validation

v0.2.7 passes 103 node tests and 141 repository tests in total, with one optional Torch smoke test skipped when Torch is unavailable. Coverage includes four-way rotation, side-level cross-component fading, harmless-tangent no-op behavior, adjacent-side smooth blending, opposite-side center completion, non-square inputs, and 1280/168 alpha-aware resizing.

A read-only visual pass over 99 production RGBA samples at `feather_width=160` activated only the same three samples containing confirmed crop evidence; the other 96 samples retained an all-one guard. This local validation is not a claim of end-to-end validation in every hosted ComfyUI environment.

## Requirements

The node uses packages normally included with ComfyUI:

- PyTorch
- NumPy
- Pillow

Mask analysis and alpha-aware resizing run on the CPU. PyTorch is used only for the ComfyUI tensor interface; CUDA and a GPU are not required.

Version: `v0.2.7`
