# ComfyUI Adaptive Edge Fade

A standalone ComfyUI custom node for repairing foregrounds clipped by the canvas edge without treating every harmless edge touch as overflow.

The node uses a precision-first crop-evidence gate, applies a curved fade only around confirmed local cut segments, and emits alpha-safe `1280 × 1280` and `168 × 168` RGBA images in one step. Wide contacts follow locally gated true-circle arcs, so their outer contours trend toward one enclosing round silhouette without applying a global circle, ellipse, or rounded-square vignette.

v0.2.5 replaces the two mirrored fields used for opposite cuts with one canvas-centred radial field. A confirmed left/right or top/bottom pair therefore follows the same circle instead of producing two incompatible arcs or rectangular feather blocks. Detection remains precision-first: unrelated edge touches never corroborate each other, and a uniform weak restore halo is rejected only on its own source run rather than suppressing genuine cuts elsewhere on the same subject.

## Features

- Uses the narrow outer probe only to discover candidates; contact alone does not trigger a fade.
- Requires a sufficiently long and dense actual-border core, stable inward support, and persistent width before classifying a segment as a crop cut.
- Keeps every unconfirmed border fragment independent. Multiple fragments and opposite-side touches cannot be merged or used to promote one another into a crop.
- Preserves natural tangencies, anti-aliased endings, shallow edge slivers, sparse border residue, and symmetric frame-filling components when crop evidence is ambiguous.
- Rejects nearly uniform low-alpha restore residue that spans almost a whole edge, while leaving stronger or differently valued cuts on the same connected subject eligible for normal detection.
- Conservatively preserves narrow border chords that widen on both sides. A second evidence band records whether the body keeps widening, closes, or reaches a deep width plateau, but cannot turn this alpha-only ambiguous shape into a confirmed crop.
- Keeps nearby and disconnected geometry unchanged, even when it lies inside another segment's geometric support.
- Processes both narrow protrusions and wide clipped bases.
- Promotes mutually unique, substantial opposite-side cuts to one shared `paired_circle_arc` field only when their robust border intervals align and both contacts lead into the same threshold-solid body.
- Fits the shared field to the thinner side's available radial depth and caps its perceptual width, preventing a thick lobe from erasing a thinner opposite handle at `168 × 168`.
- Uses a signed-distance virtual circle for wide contacts: a bottom contour stays farther outward at the center and recedes toward both sides; top, left, and right use the rotationally equivalent rule.
- Restricts every local field to its exact low-alpha 8-connected component ID.
- Uses smooth two-dimensional seed ownership when one connected component contains both a confirmed crop and a skipped edge touch, including on adjacent sides.
- Keeps narrow protrusions on compact local bowed envelopes.
- Merges adjacent sides into a local quarter-ellipse only when their alpha belongs to the same connected corner component and the pairing is unambiguous; multi-corner ties stay as rotation-equivalent independent edge fields.
- Limits tangential influence to each contact's local support; outside that support the guard is exactly `1`.
- Contracts circle depth through the tangential support, producing curved two-dimensional end caps rather than flat gray rectangles.
- Absorbs only adjacent 1–2 px anti-alias/probe tails into a confirmed run's ownership while retaining every such decision in diagnostics; substantial skipped geometry remains protected.
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

## v0.2.5 compatibility

The node class, six inputs, five outputs, names, types, order, and saved widget values are unchanged, so existing workflows load without rewiring. Diagnostics use the `clip_evidence_v7` schema and record shared-circle geometry, effective paired feather width, and precision-first skip reasons including `weak_component_halo` and `balanced_frame_flush`.

This node can soften geometry already cut by the canvas boundary; it cannot reconstruct missing off-canvas pixels. A complete flat edge that naturally ends on the last row can be pixel-identical to the same shape continuing outside the canvas. The same ambiguity exists between a complete round-shouldered tab and a cropped narrow neck after both widen into an identical deep plateau. v0.2.5 deliberately prefers a false negative over damaging such ambiguous edge-touching artwork. Exact overflow classification requires upstream overscan or a per-segment `overflow_seed_mask`.

The release was regression-tested with 92 node tests (one optional Torch smoke test skipped where Torch was unavailable), 130 repository tests in total, rotation/mirror and extreme-parameter checks, and a read-only visual pass over 99 production alpha samples at both 1280 and 168 output sizes. In that sample set, only three images with confirmed cut segments were changed; 96 edge-touching or interior cases kept an all-one edge guard.

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

Version: `v0.2.5`
