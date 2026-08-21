import json

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # 允许仓库测试在未安装 ComfyUI/Torch 的 Python 中导入纯算法。
    torch = None


NODE_VERSION = "v0.1.1"
REFERENCE_SIZE = 1280
LARGE_ICON_SIZE = 1280
SMALL_ICON_SIZE = 168


def _smootherstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _segments_from_profile(profile, merge_gap):
    indexes = np.flatnonzero(profile)
    if indexes.size == 0:
        return []

    segments = []
    start = int(indexes[0])
    previous = start
    for index in indexes[1:]:
        index = int(index)
        if index - previous > merge_gap + 1:
            segments.append((start, previous))
            start = index
        previous = index
    segments.append((start, previous))
    return segments


def _edge_profiles(alpha, scan_band, threshold):
    height, width = alpha.shape
    scan_band = max(1, min(int(scan_band), height, width))
    return {
        "top": np.max(alpha[:scan_band, :], axis=0) >= threshold,
        "right": np.max(alpha[:, width - scan_band :], axis=1) >= threshold,
        "bottom": np.max(alpha[height - scan_band :, :], axis=0) >= threshold,
        "left": np.max(alpha[:, :scan_band], axis=1) >= threshold,
    }


def _distance_to_edge_segment(height, width, side, start, end):
    """Return the exact Euclidean distance to a finite seed segment on one edge."""
    if side in ("top", "bottom"):
        x = np.arange(width, dtype=np.float32)
        lateral = np.maximum(np.maximum(float(start) - x, 0.0), x - float(end))
        if side == "top":
            inward = np.arange(height, dtype=np.float32)
        else:
            inward = np.arange(height - 1, -1, -1, dtype=np.float32)
        return np.hypot(inward[:, None], lateral[None, :])

    y = np.arange(height, dtype=np.float32)
    lateral = np.maximum(np.maximum(float(start) - y, 0.0), y - float(end))
    if side == "left":
        inward = np.arange(width, dtype=np.float32)
    else:
        inward = np.arange(width - 1, -1, -1, dtype=np.float32)
    return np.hypot(lateral[:, None], inward[None, :])


def build_adaptive_edge_guard(
    alpha,
    safety_inset=32,
    feather_width=240,
    contact_threshold=0.03,
):
    """Build a C2-continuous capsule-distance guard for every contacted edge run."""
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim != 2:
        raise ValueError("alpha 必须是二维灰度数组。")

    height, width = alpha.shape
    # Parameters are expressed in the canonical 1280 output space. Contain-resize
    # maps the source's longest side to 1280, so scale against that same side.
    scale = max(height, width) / float(REFERENCE_SIZE)
    hard_zero = max(0, int(round(float(safety_inset) * scale)))
    base_feather = max(1, int(round(float(feather_width) * scale)))
    scan_band = max(1, int(round(48 * scale)))
    merge_gap = max(1, int(round(12 * scale)))
    guard = np.ones((height, width), dtype=np.float32)
    contacts = []

    profiles = _edge_profiles(alpha, scan_band, float(contact_threshold))
    for side in ("top", "right", "bottom", "left"):
        side_length = width if side in ("top", "bottom") else height
        for start, end in _segments_from_profile(profiles[side], merge_gap):
            length = end - start + 1
            fraction = length / float(side_length)
            # 宽触边同样必须处理，并使用更深的向内渐隐；绝不作为跳过条件。
            is_wide = fraction >= 0.45
            local_feather = max(1, int(round(base_feather * (1.2 if is_wide else 1.0))))
            gamma = 1.6

            distance = _distance_to_edge_segment(height, width, side, start, end)
            transition = (distance - float(hard_zero)) / float(local_feather)
            local_guard = np.power(_smootherstep(transition), gamma).astype(np.float32)
            guard = np.minimum(guard, local_guard)
            contacts.append(
                {
                    "side": side,
                    "start": start,
                    "end": end,
                    "length": length,
                    "fraction": round(fraction, 6),
                    "wide": is_wide,
                    "hard_zero_px": hard_zero,
                    "feather_px": local_feather,
                    "gamma": gamma,
                }
            )

    return guard, contacts


def apply_adaptive_edge_fade(
    rgb,
    alpha,
    safety_inset=32,
    feather_width=240,
    contact_threshold=0.03,
):
    rgb = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb 必须是 HWC RGB/RGBA 数组。")
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("final_alpha 与 rgba 的高宽必须一致。")

    rgb = np.clip(rgb[..., :3], 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    guard, contacts = build_adaptive_edge_guard(
        alpha,
        safety_inset=safety_inset,
        feather_width=feather_width,
        contact_threshold=contact_threshold,
    )
    output_alpha = alpha * guard
    return rgb, output_alpha.astype(np.float32), guard, contacts


def _srgb_to_linear(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _resize_float_channel(channel, width, height, resample):
    image = Image.fromarray(np.asarray(channel, dtype=np.float32))
    resized = image.resize((int(width), int(height)), resample=resample)
    return np.asarray(resized, dtype=np.float32)


def _fit_float_channels(channels, target_size):
    """Resize HWC float channels with one filter, then zero-pad instead of edge replication."""
    height, width, _ = channels.shape
    scale = min(target_size / float(width), target_size / float(height))
    resized_width = max(1, min(target_size, int(round(width * scale))))
    resized_height = max(1, min(target_size, int(round(height * scale))))

    if resized_width == width and resized_height == height:
        resized = channels.astype(np.float32, copy=True)
    else:
        downsample = resized_width < width or resized_height < height
        resample = Image.Resampling.LANCZOS if downsample else Image.Resampling.BICUBIC
        resized = np.stack(
            [
                _resize_float_channel(channels[..., index], resized_width, resized_height, resample)
                for index in range(channels.shape[2])
            ],
            axis=-1,
        )

    output = np.zeros((target_size, target_size, channels.shape[2]), dtype=np.float32)
    offset_x = (target_size - resized_width) // 2
    offset_y = (target_size - resized_height) // 2
    output[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
        :,
    ] = resized
    return output


def alpha_aware_square_resize(rgb, alpha, target_size):
    """Return an unsanitized straight-alpha master after premultiplied linear-light resize."""
    rgb = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    linear_rgb = _srgb_to_linear(rgb)
    premultiplied = linear_rgb * alpha[..., None]
    packed = np.concatenate([premultiplied, alpha[..., None]], axis=-1)
    resized = np.clip(_fit_float_channels(packed, int(target_size)), 0.0, 1.0)

    output_alpha = resized[..., 3]
    output_linear = np.zeros_like(resized[..., :3])
    valid = output_alpha > 1.0e-6
    output_linear[valid] = resized[..., :3][valid] / output_alpha[valid, None]
    output_rgb = _linear_to_srgb(output_linear)
    return np.concatenate([output_rgb, output_alpha[..., None]], axis=-1).astype(np.float32)


def sanitize_low_alpha_rgb(rgba, cleanup_limit=0.06):
    """Suppress straight-alpha RGB residues once, without changing nonzero alpha itself."""
    output = np.clip(np.asarray(rgba, dtype=np.float32), 0.0, 1.0).copy()
    alpha = output[..., 3]
    cleanup_limit = max(0.0, float(cleanup_limit))
    if cleanup_limit > 0:
        cleanup_weight = _smootherstep(alpha / cleanup_limit)
        output[..., :3] *= cleanup_weight[..., None]
    transparent = alpha < (0.5 / 255.0)
    output[..., 3][transparent] = 0.0
    output[..., :3][transparent] = 0.0
    return output.astype(np.float32)


def _normalize_image_batch(value):
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] < 3:
        raise ValueError("rgba 必须是标准 ComfyUI BHWC IMAGE，且至少包含 RGB 三通道。")
    return array


def _normalize_mask_batch(value, batch_size):
    array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError("final_alpha 必须是标准 ComfyUI BHW MASK。")
    if array.shape[0] == 1 and batch_size > 1:
        array = np.repeat(array, batch_size, axis=0)
    if array.shape[0] != batch_size:
        raise ValueError("rgba 与 final_alpha 的 batch 数不一致。")
    return array


def combine_visible_alpha(image, visible_alpha):
    """Use 1=visible mask semantics and never revive pixels hidden by RGBA alpha."""
    image = np.asarray(image, dtype=np.float32)
    visible_alpha = np.clip(np.asarray(visible_alpha, dtype=np.float32), 0.0, 1.0)
    if image.shape[:2] != visible_alpha.shape:
        raise ValueError("final_alpha 与 rgba 的高宽必须一致；请连接同一上游节点的同尺寸输出。")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgba 必须是 HWC RGB/RGBA 数组。")
    if image.shape[2] >= 4:
        return np.minimum(visible_alpha, np.clip(image[..., 3], 0.0, 1.0))
    return visible_alpha


class GameUGCAdaptiveEdgeFade:
    """Repair clipped canvas contacts and emit alpha-safe 1280/168 RGBA icons."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgba": ("IMAGE",),
                "final_alpha": (
                    "MASK",
                    {
                        "tooltip": "可见 Alpha：1=可见、0=透明；直接连接上游可见 Alpha，无需反相。"
                    },
                ),
                "safety_inset": (
                    "INT",
                    {"default": 32, "min": 0, "max": 256, "step": 1},
                ),
                "feather_width": (
                    "INT",
                    {"default": 240, "min": 8, "max": 640, "step": 1},
                ),
                "contact_threshold": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001},
                ),
                "rgb_cleanup_alpha": (
                    "FLOAT",
                    {"default": 0.06, "min": 0.0, "max": 0.25, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = (
        "icon_1280_rgba",
        "icon_168_rgba",
        "final_alpha_1280",
        "edge_guard_1280",
        "diagnostics",
    )
    FUNCTION = "repair"
    CATEGORY = "image/alpha"

    def repair(
        self,
        rgba,
        final_alpha,
        safety_inset=32,
        feather_width=240,
        contact_threshold=0.03,
        rgb_cleanup_alpha=0.06,
    ):
        if torch is None:
            raise RuntimeError("GameUGCAdaptiveEdgeFade 必须在已安装 PyTorch 的 ComfyUI 中运行。")

        image_batch = _normalize_image_batch(rgba)
        mask_batch = _normalize_mask_batch(final_alpha, image_batch.shape[0])
        large_batch = []
        small_batch = []
        guard_batch = []
        diagnostics = []

        for batch_index, image in enumerate(image_batch):
            alpha = combine_visible_alpha(image, mask_batch[batch_index])
            rgb, repaired_alpha, guard, contacts = apply_adaptive_edge_fade(
                image[..., :3],
                alpha,
                safety_inset=safety_inset,
                feather_width=feather_width,
                contact_threshold=contact_threshold,
            )
            master_1280 = alpha_aware_square_resize(rgb, repaired_alpha, LARGE_ICON_SIZE)
            master_168 = alpha_aware_square_resize(
                master_1280[..., :3],
                master_1280[..., 3],
                SMALL_ICON_SIZE,
            )
            icon_1280 = sanitize_low_alpha_rgb(master_1280, rgb_cleanup_alpha)
            icon_168 = sanitize_low_alpha_rgb(master_168, rgb_cleanup_alpha)
            guard_1280 = _fit_float_channels(guard[..., None], LARGE_ICON_SIZE)[..., 0]

            large_batch.append(icon_1280)
            small_batch.append(icon_168)
            guard_batch.append(np.clip(guard_1280, 0.0, 1.0))
            diagnostics.append(
                {
                    "batch": batch_index,
                    "node_version": NODE_VERSION,
                    "input_size": [int(image.shape[1]), int(image.shape[0])],
                    "contacts": contacts,
                }
            )

        large_array = np.stack(large_batch, axis=0).astype(np.float32)
        small_array = np.stack(small_batch, axis=0).astype(np.float32)
        guard_array = np.stack(guard_batch, axis=0).astype(np.float32)
        alpha_array = large_array[..., 3].astype(np.float32)
        return (
            torch.from_numpy(large_array),
            torch.from_numpy(small_array),
            torch.from_numpy(alpha_array),
            torch.from_numpy(guard_array),
            json.dumps(diagnostics, ensure_ascii=False, separators=(",", ":")),
        )


NODE_CLASS_MAPPINGS = {
    "GameUGCAdaptiveEdgeFade": GameUGCAdaptiveEdgeFade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GameUGCAdaptiveEdgeFade": "Adaptive Edge Fade + Alpha-safe 1280/168",
}
