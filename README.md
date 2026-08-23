# ComfyUI Adaptive Edge Fade

A standalone ComfyUI custom node for repairing foregrounds clipped by the canvas edge without treating every harmless edge touch as overflow.

The node uses a precision-first crop-evidence gate, applies a curved fade only around confirmed local cut segments, and emits alpha-safe `1280 × 1280` and `168 × 168` RGBA images in one step. Wide contacts follow locally gated true-circle arcs, so their outer contours trend toward one enclosing round silhouette without applying a global circle, ellipse, or rounded-square vignette.

## Features

- Uses the narrow outer probe only to discover candidates; contact alone does not trigger a fade.
- Requires a sufficiently long and dense actual-border core, stable inward support, and persistent width before classifying a segment as a crop cut.
- Preserves natural tangencies, anti-aliased endings, shallow edge slivers, sparse border residue, and symmetric frame-filling components when crop evidence is ambiguous.
- Keeps nearby and disconnected geometry unchanged, even when it lies inside another segment's geometric support.
- Processes both narrow protrusions and wide clipped bases.
- Uses a signed-distance virtual circle for wide contacts: a bottom contour stays farther outward at the center and recedes toward both sides; top, left, and right use the rotationally equivalent rule.
- Restricts every local field to its exact low-alpha 8-connected component ID.
- Uses smooth two-dimensional seed ownership when one connected component contains both a confirmed crop and a skipped edge touch, including on adjacent sides.
- Keeps narrow protrusions on compact local bowed envelopes.
- Merges adjacent sides into a local quarter-ellipse only when their alpha belongs to the same connected corner component and the pairing is unambiguous; multi-corner ties stay as rotation-equivalent independent edge fields.
- Limits tangential influence to each contact's local support; outside that support the guard is exactly `1`.
- Combines local guards smoothly without hard rectangular projections or `min` seams.
- Resizes linear-light premultiplied RGB and alpha together.
- Uses transparent contain-padding instead of edge-pixel replication.
- Cleans only very-low-alpha RGB residues without globally clipping soft alpha effects.
- No model, network request, telemetry, or external asset is required.

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
- `safety_inset`: fully transparent inset across each confirmed contact run, expressed in the canonical 1280 output space.
- `feather_width`: inward feather distance in the canonical 1280 output space.
- `contact_threshold`: alpha threshold used to estimate the deeper body span. The outer probe intentionally uses a much lower half-8-bit-alpha threshold only for candidate discovery; the actual border still has to pass the solid-core, inward-support, and width-persistence evidence gates.
- `rgb_cleanup_alpha`: low-alpha range in which residual straight RGB is attenuated.

If `rgba` already has an alpha channel, the node intersects it with `final_alpha`, so pixels hidden by either source cannot be revived.

## v0.2.2 compatibility

The node class, six inputs, five outputs, names, types, order, and saved widget values are unchanged, so existing workflows load without rewiring. v0.2.2 adds conservative crop-evidence classification, per-component decisions, two-dimensional confirmed/skipped ownership, and rotation-stable corner pairing. Diagnostics use the `clip_evidence_v4` schema and report both confirmed contacts and skipped candidates.

This node can soften geometry already cut by the canvas boundary; it cannot reconstruct missing off-canvas pixels. A complete flat edge that naturally ends on the last row can be pixel-identical to the same shape continuing outside the canvas, so no post-crop alpha-only algorithm can distinguish every such case perfectly. v0.2.2 deliberately prefers a false negative over damaging ambiguous edge-touching artwork. Exact overflow classification requires upstream overscan or a per-segment `overflow_seed_mask`.

## Outputs

- `icon_1280_rgba`
- `icon_168_rgba`
- `final_alpha_1280`
- `edge_guard_1280`
- `diagnostics`

## Requirements

The node uses packages normally included with ComfyUI:

- PyTorch
- NumPy
- Pillow

Version: `v0.2.2`
