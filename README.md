# ComfyUI Adaptive Edge Fade

A standalone ComfyUI custom node for repairing foregrounds clipped by the canvas edge without treating every harmless edge touch as overflow.

The node uses a precision-first crop-evidence gate, applies a curved fade only around confirmed cut segments, and emits alpha-safe `1280 × 1280` and `168 × 168` RGBA images in one step. Wide contacts follow component-gated true-circle transition bands, so their outer contours trend toward one enclosing round silhouette without applying a global circle, ellipse, or rounded-square vignette.

v0.2.6 removes the fixed tangential window that could split a low hand or fur branch from a clipped torso. For a confirmed wide cut—and for a medium cut only when a meaningful low branch proves it is needed—the node first selects the complete 8-connected subject, then intersects that subject with the near-edge circle transition band. A low branch connected through a higher shoulder therefore shares the same smooth fade, while high geometry, completed-circle interior geometry, disconnected objects, and skipped edge touches remain unchanged.

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
- Uses a continuous two-dimensional safety-capsule ownership field when one connected component contains both a confirmed crop and a skipped edge touch. The confirmed run plus its inset forms a capsule with rounded ends; distance-ratio blending protects skipped geometry without creating a rectangular ribbon, vertical wall, or hard winner boundary.
- Selects the complete seeded subject before intersecting it with a full-canvas circle transition band, allowing low branches connected through higher geometry to fade without a vertical support seam.
- Automatically upgrades a medium contact only when the same component contains a meaningful low two-dimensional branch outside the former local support; ordinary medium and narrow contacts stay local.
- Forces the circle field back to exact guard `1` after its completion depth, so long non-square subjects cannot re-darken on the far side of the virtual circle.
- Applies one circle field per side/component group even when the confirmed boundary contains multiple runs; only the actual source runs receive the fully transparent safety inset and unconfirmed gaps stay untouched.
- Keeps narrow protrusions on compact local bowed envelopes.
- Merges adjacent sides into a local quarter-ellipse only when their alpha belongs to the same connected corner component and the pairing is unambiguous; multi-corner ties stay as rotation-equivalent independent edge fields.
- Avoids one-dimensional tangential ownership on single-edge circle fields. Confirmed literal/core pixels remain exactly transparent, skipped literal pixels remain exactly protected, and the region between them transitions continuously in two-dimensional distance.
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

## v0.2.6 compatibility

The node class, six inputs, five outputs, names, types, order, and saved widget values are unchanged, so existing workflows load without rewiring. Diagnostics use the `clip_evidence_v8` schema and additionally record seeded full-circle scope, completion depth, source-run count, and transition-band pixels, while retaining shared-circle geometry and precision-first skip reasons such as `weak_component_halo` and `balanced_frame_flush`.

This node can soften geometry already cut by the canvas boundary; it cannot reconstruct missing off-canvas pixels. A complete flat edge that naturally ends on the last row can be pixel-identical to the same shape continuing outside the canvas. The same ambiguity exists between a complete round-shouldered tab and a cropped narrow neck after both widen into an identical deep plateau. v0.2.6 deliberately prefers a false negative over damaging such ambiguous edge-touching artwork. Exact overflow classification requires upstream overscan or a per-segment `overflow_seed_mask`.

The release was regression-tested with 98 node tests and 136 repository tests in total (including one optional Torch smoke test skipped where Torch was unavailable), rotation/mirror and extreme-parameter checks, and a read-only visual pass over 99 production alpha samples at both 1280 and 168 output sizes. Only three images contained confirmed cut segments; 96 edge-touching or interior cases kept an all-one edge guard. Compared with v0.2.5, 98 samples were pixel-identical and the one changed multi-edge sample gained the intended whole-component circle transition.

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

Mask analysis and alpha-aware resizing run on the CPU. PyTorch is used only for the ComfyUI tensor interface; CUDA and a GPU are not required by this node.

Version: `v0.2.6`
