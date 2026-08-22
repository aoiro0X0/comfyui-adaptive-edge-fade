# ComfyUI Adaptive Edge Fade

A standalone ComfyUI custom node for repairing foregrounds clipped by the canvas edge.

The node confirms actual outer-edge contact, applies a curved fade only around each contacted component, and emits alpha-safe `1280 × 1280` and `168 × 168` RGBA images in one step. It does not apply a global circle, ellipse, or rounded-square vignette.

## Features

- Confirms contact in a narrow outer probe before changing any alpha.
- Keeps nearby geometry that does not touch the outer probe unchanged.
- Processes both narrow protrusions and wide clipped bases.
- Uses per-contact bowed depth envelopes, so wide fades curve inward instead of forming flat rectangular strips.
- Merges adjacent sides into a local quarter-ellipse only when their alpha belongs to the same connected corner component.
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
- `contact_threshold`: alpha threshold used to estimate the deeper body span after contact is confirmed. The outer probe intentionally uses a much lower half-8-bit-alpha threshold so faint edge residue cannot become a bright straight line after preview or resize.
- `rgb_cleanup_alpha`: low-alpha range in which residual straight RGB is attenuated.

If `rgba` already has an alpha channel, the node intersects it with `final_alpha`, so pixels hidden by either source cannot be revived.

## v0.2.0 compatibility

The node class, six inputs, and five outputs keep the same names, types, and order as v0.1.1, so existing workflows still load. Saved widget values are also preserved; v0.2.0 changes the visual behavior and uses `8` as the new default `safety_inset`, but an older workflow saved with `32` will continue to use `32` until changed.

This node can soften geometry already cut by the canvas boundary; it cannot reconstruct missing off-canvas pixels. If the complete shape must remain visible, scale the subject down or extend the image upstream before matting.

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

Version: `v0.2.0`
