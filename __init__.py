import json
from collections import deque

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # 允许仓库测试在未安装 ComfyUI/Torch 的 Python 中导入纯算法。
    torch = None


NODE_VERSION = "v0.2.2"
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
    vertical_band = max(1, min(int(scan_band), height))
    horizontal_band = max(1, min(int(scan_band), width))
    return {
        "top": np.max(alpha[:vertical_band, :], axis=0) >= threshold,
        "right": np.max(alpha[:, width - horizontal_band :], axis=1) >= threshold,
        "bottom": np.max(alpha[height - vertical_band :, :], axis=0) >= threshold,
        "left": np.max(alpha[:, :horizontal_band], axis=1) >= threshold,
    }


def _boolean_iou(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.count_nonzero(first | second)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first & second)) / float(union)


def _oriented_full_alpha(alpha, side):
    """Return the full canvas as (inward distance, lateral coordinate)."""
    if side == "top":
        return alpha
    if side == "bottom":
        return alpha[::-1, :]
    if side == "left":
        return alpha.T
    return alpha[:, ::-1].T


def _label_8_connected_components(visible):
    """Label a binary mask with 8-connectivity using row runs."""
    visible = np.asarray(visible, dtype=bool)
    height, width = visible.shape
    parent = []
    rank = []
    runs = []

    def make_set():
        identifier = len(parent)
        parent.append(identifier)
        rank.append(0)
        return identifier

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    previous = []
    for y in range(height):
        padded = np.pad(visible[y].astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        current = []
        previous_index = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            identifier = make_set()
            while (
                previous_index < len(previous)
                and previous[previous_index][1] < start - 1
            ):
                previous_index += 1
            overlap_index = previous_index
            while (
                overlap_index < len(previous)
                and previous[overlap_index][0] <= end + 1
            ):
                union(identifier, previous[overlap_index][2])
                overlap_index += 1
            current.append((int(start), int(end), identifier))
            runs.append((y, int(start), int(end), identifier))
        previous = current

    labels = np.zeros((height, width), dtype=np.int32)
    root_to_label = {}
    for y, start, end, identifier in runs:
        root = find(identifier)
        label = root_to_label.setdefault(root, len(root_to_label) + 1)
        labels[y, start : end + 1] = label
    return labels


def _edge_candidates(
    alpha,
    labels,
    side,
    edge_probe,
    scan_band,
    outer_threshold,
    span_threshold,
    max_extension,
):
    """Return independent actual-border runs, grouped by low-alpha component."""
    oriented_alpha = _oriented_full_alpha(alpha, side)
    oriented_labels = _oriented_full_alpha(labels, side)
    normal_length, side_length = oriented_alpha.shape
    probe_depth = max(1, min(int(edge_probe), normal_length))
    evidence_depth = max(1, min(int(scan_band), normal_length))
    component_ids = np.unique(oriented_labels[:probe_depth])
    component_ids = component_ids[component_ids > 0]
    candidates = []

    for component_id in component_ids.tolist():
        probe_profile = np.any(
            oriented_labels[:probe_depth] == component_id,
            axis=0,
        )
        actual_profile = (
            (oriented_labels[0] == component_id)
            & (oriented_alpha[0] >= float(outer_threshold))
        )
        actual_runs = _segments_from_profile(actual_profile, 0)
        seed_runs = actual_runs or _segments_from_profile(probe_profile, 0)

        deep_profile = np.any(
            (oriented_labels[:evidence_depth] == component_id)
            & (oriented_alpha[:evidence_depth] >= float(span_threshold)),
            axis=0,
        )
        deep_runs = _segments_from_profile(deep_profile, 0)
        for seed_start, seed_end in seed_runs:
            matches = [
                (deep_start, deep_end)
                for deep_start, deep_end in deep_runs
                if deep_start <= seed_end + max_extension
                and deep_end >= seed_start - max_extension
            ]
            if matches:
                span_start = max(
                    min(start for start, _ in matches),
                    seed_start - max_extension,
                )
                span_end = min(
                    max(end for _, end in matches),
                    seed_end + max_extension,
                )
            else:
                span_start, span_end = seed_start, seed_end
            candidates.append(
                {
                    "side": side,
                    "component_id": int(component_id),
                    "seed_start": int(seed_start),
                    "seed_end": int(seed_end),
                    "span_start": int(max(0, span_start)),
                    "span_end": int(min(side_length - 1, span_end)),
                    "actual_border": bool(actual_runs),
                }
            )
    return candidates


def _edge_segment_clip_evidence(
    alpha,
    labels,
    candidate,
    edge_probe,
    scan_band,
    outer_threshold,
    span_threshold,
    scale,
):
    """Conservatively distinguish a stable crop cut from a grazing touch.

    The outer low-alpha band only locates a candidate.  Confirmation comes from
    the actual canvas-border row: it must contain a sufficiently long, locally
    solid run whose width persists into the canvas.  This deliberately rejects
    anti-aliased tangencies and very shallow/ambiguous contacts.
    """
    side = candidate["side"]
    oriented = _oriented_full_alpha(alpha, side)
    oriented_labels = _oriented_full_alpha(labels, side)
    normal_length, side_length = oriented.shape
    depth = max(1, min(int(scan_band), normal_length))
    span_start = max(0, min(side_length - 1, int(candidate["span_start"])))
    span_end = max(span_start, min(side_length - 1, int(candidate["span_end"])))
    seed_start = max(span_start, int(candidate["seed_start"]))
    seed_end = min(span_end, int(candidate["seed_end"]))
    patch = oriented[:depth, span_start : span_end + 1]
    component = (
        oriented_labels[:depth, span_start : span_end + 1]
        == int(candidate["component_id"])
    )

    border_alpha = patch[0]
    seed_profile = np.zeros(patch.shape[1], dtype=bool)
    seed_profile[
        seed_start - span_start : seed_end - span_start + 1
    ] = True
    low_border = (
        component[0]
        & seed_profile
        & (border_alpha >= float(outer_threshold))
    )
    actual_border_pixels = int(np.count_nonzero(low_border))
    metrics = {
        "decision": "skipped",
        "reason": "actual_border_empty",
        "actual_border_pixels": actual_border_pixels,
        "robust_border_run_px": 0,
        "robust_core_pixels": 0,
        "robust_hull_coverage": 0.0,
        "border_fill_p25": 0.0,
        "border_robust_fraction": 0.0,
        "inward_width_p90_px": 0.0,
        "inward_support_depth_px": 0,
        "inward_support_required_px": 0,
        "inward_support_prefix_coverage": 0.0,
        "unexplained_inward_width_p90_px": 0.0,
        "weak_left_tail_px": 0,
        "weak_right_tail_px": 0,
        "left_expansion_p90_px": 0.0,
        "right_expansion_p90_px": 0.0,
        "width_persistence": 0.0,
        "evidence_depth_px": int(depth),
    }
    if actual_border_pixels == 0:
        return False, metrics

    peak = np.max(np.where(component, patch, 0.0), axis=0)
    has_core = low_border & (peak >= float(span_threshold))
    if not np.any(has_core):
        metrics["reason"] = "weak_edge_only"
        return False, metrics

    relative_edge = np.zeros_like(border_alpha, dtype=np.float32)
    relative_edge[has_core] = border_alpha[has_core] / np.maximum(
        peak[has_core], 1.0e-8
    )
    border_fill_p25 = float(np.percentile(relative_edge[has_core], 25))
    robust = has_core & (relative_edge >= 0.85)
    robust_merge_gap = max(1, int(round(2.0 * float(scale))))
    robust_segments = _segments_from_profile(robust, robust_merge_gap)
    robust_start, robust_end = max(
        robust_segments,
        key=lambda segment: segment[1] - segment[0] + 1,
        default=(0, -1),
    )
    robust_run = max(0, int(robust_end - robust_start + 1))
    robust_core_pixels = (
        int(np.count_nonzero(robust[robust_start : robust_end + 1]))
        if robust_run > 0
        else 0
    )
    robust_hull_coverage = (
        float(robust_core_pixels) / float(robust_run)
        if robust_run > 0
        else 0.0
    )
    robust_fraction = float(np.count_nonzero(robust)) / float(actual_border_pixels)

    core = component & (patch >= float(span_threshold))
    inward_start = min(depth - 1, max(1, int(edge_probe)))
    inward_widths = np.count_nonzero(core[inward_start:], axis=1)
    positive_widths = inward_widths[inward_widths > 0]
    inward_width_p90 = (
        float(np.percentile(positive_widths, 90))
        if positive_widths.size
        else 0.0
    )
    left_expansions = []
    right_expansions = []
    if robust_run > 0:
        for row in core[inward_start:]:
            indexes = np.flatnonzero(row)
            if indexes.size == 0:
                continue
            left_expansions.append(max(0, robust_start - int(indexes[0])))
            right_expansions.append(max(0, int(indexes[-1]) - robust_end))
    left_expansion_p90 = (
        float(np.percentile(left_expansions, 90))
        if left_expansions
        else 0.0
    )
    right_expansion_p90 = (
        float(np.percentile(right_expansions, 90))
        if right_expansions
        else 0.0
    )
    low_border_indexes = np.flatnonzero(low_border)
    weak_left_tail = (
        max(0, robust_start - int(low_border_indexes[0]))
        if low_border_indexes.size and robust_run > 0
        else 0
    )
    weak_right_tail = (
        max(0, int(low_border_indexes[-1]) - robust_end)
        if low_border_indexes.size and robust_run > 0
        else 0
    )
    unexplained_left = max(0.0, left_expansion_p90 - weak_left_tail)
    unexplained_right = max(0.0, right_expansion_p90 - weak_right_tail)
    unexplained_inward_width = (
        float(robust_run) + unexplained_left + unexplained_right
    )
    persistence = (
        min(1.0, float(robust_run) / unexplained_inward_width)
        if unexplained_inward_width > 0.0
        else 0.0
    )
    minimum_support_depth = min(
        depth,
        max(int(edge_probe) + 1, int(round(12.0 * float(scale)))),
    )
    if robust_run > 0:
        support_rows = np.any(
            core[:, robust_start : robust_end + 1],
            axis=1,
        )
    else:
        support_rows = np.zeros(depth, dtype=bool)
    support_indexes = np.flatnonzero(support_rows)
    inward_support_depth = (
        int(support_indexes[-1]) + 1 if support_indexes.size else 0
    )
    support_prefix = support_rows[:minimum_support_depth]
    support_prefix_coverage = (
        float(np.count_nonzero(support_prefix))
        / float(minimum_support_depth)
        if minimum_support_depth > 0
        else 0.0
    )

    metrics.update(
        {
            "robust_border_run_px": robust_run,
            "robust_core_pixels": robust_core_pixels,
            "robust_hull_coverage": round(robust_hull_coverage, 6),
            "border_fill_p25": round(border_fill_p25, 6),
            "border_robust_fraction": round(robust_fraction, 6),
            "inward_width_p90_px": round(inward_width_p90, 3),
            "inward_support_depth_px": int(inward_support_depth),
            "inward_support_required_px": int(minimum_support_depth),
            "inward_support_prefix_coverage": round(
                support_prefix_coverage, 6
            ),
            "unexplained_inward_width_p90_px": round(
                unexplained_inward_width, 3
            ),
            "weak_left_tail_px": int(weak_left_tail),
            "weak_right_tail_px": int(weak_right_tail),
            "left_expansion_p90_px": round(left_expansion_p90, 3),
            "right_expansion_p90_px": round(right_expansion_p90, 3),
            "width_persistence": round(persistence, 6),
        }
    )
    minimum_run = max(2, int(round(8.0 * float(scale))))
    if robust_run == 0:
        metrics["reason"] = "antialiased_touch"
        return False, metrics
    if robust_run < minimum_run:
        metrics["reason"] = "grazing_tangent"
        return False, metrics
    if robust_core_pixels < minimum_run or robust_hull_coverage < 0.75:
        metrics["reason"] = "fragmented_border_core"
        return False, metrics
    if (
        inward_width_p90 <= 0.0
        or inward_support_depth < minimum_support_depth
        or not bool(support_rows[minimum_support_depth - 1])
        or support_prefix_coverage < 0.75
    ):
        metrics["reason"] = "shallow_ambiguous"
        return False, metrics
    if persistence < 0.55:
        metrics["reason"] = "tangent_expansion"
        return False, metrics

    metrics["decision"] = "confirmed_clip"
    metrics["reason"] = "stable_border_cut"
    return True, metrics


def _balanced_frame_flush_evidence(
    alpha,
    labels,
    candidates_by_side,
    edge_probe,
    span_threshold,
):
    """Return per-component frame-filling exemptions.

    This is intentionally a conservative false-positive guard.  A perfectly
    symmetric oversized object could look identical after cropping, so such a
    case is skipped unless a future upstream node supplies out-of-canvas data.
    """
    candidates_by_component = {}
    for side, candidates in candidates_by_side.items():
        for candidate in candidates:
            component_id = int(candidate["component_id"])
            candidates_by_component.setdefault(component_id, {}).setdefault(
                side, []
            ).append(candidate)

    exemptions = {}
    height, width = alpha.shape
    probe = max(1, int(edge_probe))
    for component_id, component_candidates in candidates_by_component.items():
        candidate_sides = set(component_candidates)
        has_left_right = {"left", "right"}.issubset(candidate_sides)
        has_top_bottom = {"top", "bottom"}.issubset(candidate_sides)
        if len(candidate_sides) < 3 or not (has_left_right or has_top_bottom):
            continue

        component = labels == component_id
        if not np.any(component):
            continue
        component_peak = float(np.max(alpha[component]))
        if component_peak < float(span_threshold):
            continue
        frame_threshold = max(
            float(span_threshold),
            0.25 * component_peak,
        )
        body = component & (alpha >= frame_threshold)
        rows = np.flatnonzero(np.any(body, axis=1))
        columns = np.flatnonzero(np.any(body, axis=0))
        if rows.size == 0 or columns.size == 0:
            continue
        y_min, y_max = int(rows[0]), int(rows[-1])
        x_min, x_max = int(columns[0]), int(columns[-1])
        bbox_fill_x = float(x_max - x_min + 1) / float(width)
        bbox_fill_y = float(y_max - y_min + 1) / float(height)
        fills_x = x_min <= probe and width - 1 - x_max <= probe
        fills_y = y_min <= probe and height - 1 - y_max <= probe
        if not (
            fills_x
            and fills_y
            and bbox_fill_x >= 0.985
            and bbox_fill_y >= 0.985
        ):
            continue

        horizontal_mirror_iou = _boolean_iou(body, body[:, ::-1])
        vertical_mirror_iou = _boolean_iou(body, body[::-1, :])
        left_profile = np.max(body[:, :probe], axis=1)
        right_profile = np.max(body[:, width - probe :], axis=1)
        left_right_iou = _boolean_iou(left_profile, right_profile)
        top_profile = np.max(body[:probe, :], axis=0)
        bottom_profile = np.max(body[height - probe :, :], axis=0)
        top_bottom_iou = _boolean_iou(top_profile, bottom_profile)
        horizontal_symmetries = []
        if "top" in candidate_sides:
            horizontal_symmetries.append(
                _boolean_iou(top_profile, top_profile[::-1])
            )
        if "bottom" in candidate_sides:
            horizontal_symmetries.append(
                _boolean_iou(bottom_profile, bottom_profile[::-1])
            )
        horizontal_edge_symmetry = (
            min(horizontal_symmetries) if horizontal_symmetries else 0.0
        )
        vertical_symmetries = []
        if "left" in candidate_sides:
            vertical_symmetries.append(
                _boolean_iou(left_profile, left_profile[::-1])
            )
        if "right" in candidate_sides:
            vertical_symmetries.append(
                _boolean_iou(right_profile, right_profile[::-1])
            )
        vertical_edge_symmetry = (
            min(vertical_symmetries) if vertical_symmetries else 0.0
        )
        horizontal_match = (
            has_left_right
            and horizontal_mirror_iou >= 0.76
            and left_right_iou >= 0.80
            and horizontal_edge_symmetry >= 0.72
        )
        vertical_match = (
            has_top_bottom
            and vertical_mirror_iou >= 0.76
            and top_bottom_iou >= 0.80
            and vertical_edge_symmetry >= 0.72
        )
        if not (horizontal_match or vertical_match):
            continue

        if horizontal_match and vertical_match:
            symmetry_axis = "both"
        elif horizontal_match:
            symmetry_axis = "horizontal"
        else:
            symmetry_axis = "vertical"
        exemptions[component_id] = {
            "decision": "skipped",
            "reason": "balanced_frame_flush",
            "component_id": int(component_id),
            "candidate_sides": sorted(candidate_sides),
            "component_peak_alpha": round(component_peak, 6),
            "frame_threshold": round(frame_threshold, 6),
            "bbox_fill_x": round(bbox_fill_x, 6),
            "bbox_fill_y": round(bbox_fill_y, 6),
            "symmetry_axis": symmetry_axis,
            "horizontal_mirror_iou": round(horizontal_mirror_iou, 6),
            "vertical_mirror_iou": round(vertical_mirror_iou, 6),
            "left_right_edge_iou": round(left_right_iou, 6),
            "top_bottom_edge_iou": round(top_bottom_iou, 6),
            "horizontal_edge_symmetry": round(horizontal_edge_symmetry, 6),
            "vertical_edge_symmetry": round(vertical_edge_symmetry, 6),
        }
    return exemptions


def _wide_multiplier(fraction):
    wide_mix = _smootherstep((float(fraction) - 0.35) / 0.45)
    return 1.0 + 0.15 * float(wide_mix)


def _interval_distance(coordinates, start, end):
    return np.maximum(
        np.maximum(float(start) - coordinates, coordinates - float(end)),
        0.0,
    )


def _lateral_ownership_profile(
    side_length,
    confirmed_start,
    confirmed_end,
    skipped_ranges,
    blend_width,
):
    """Softly reserve nearby skipped runs on the same connected component."""
    if not skipped_ranges:
        return np.ones(int(side_length), dtype=np.float32)
    coordinates = np.arange(int(side_length), dtype=np.float32)
    confirmed_distance = _interval_distance(
        coordinates, confirmed_start, confirmed_end
    )
    skipped_distance = np.full(int(side_length), np.inf, dtype=np.float32)
    for skipped_start, skipped_end in skipped_ranges:
        skipped_distance = np.minimum(
            skipped_distance,
            _interval_distance(coordinates, skipped_start, skipped_end),
        )
    blend_width = max(1.0, float(blend_width))
    ownership = _smootherstep(
        0.5 + 0.5 * (skipped_distance - confirmed_distance) / blend_width
    )
    for skipped_start, skipped_end in skipped_ranges:
        ownership[int(skipped_start) : int(skipped_end) + 1] = 0.0
    ownership[int(confirmed_start) : int(confirmed_end) + 1] = 1.0
    return ownership.astype(np.float32)


def _edge_seed_distance(shape, side, start, end):
    """Euclidean distance to one segment on the actual canvas boundary."""
    height, width = shape
    y, x = np.ogrid[:height, :width]
    if side == "top":
        normal = y.astype(np.float32)
        lateral = _interval_distance(x.astype(np.float32), start, end)
    elif side == "bottom":
        normal = (height - 1 - y).astype(np.float32)
        lateral = _interval_distance(x.astype(np.float32), start, end)
    elif side == "left":
        normal = x.astype(np.float32)
        lateral = _interval_distance(y.astype(np.float32), start, end)
    else:
        normal = (width - 1 - x).astype(np.float32)
        lateral = _interval_distance(y.astype(np.float32), start, end)
    return np.hypot(normal, lateral).astype(np.float32)


def _write_edge_seed(field, side, start, end, value):
    height, width = field.shape
    if side in ("top", "bottom"):
        start = max(0, min(width - 1, int(start)))
        end = max(start, min(width - 1, int(end)))
        row = 0 if side == "top" else height - 1
        field[row, start : end + 1] = value
    else:
        start = max(0, min(height - 1, int(start)))
        end = max(start, min(height - 1, int(end)))
        column = 0 if side == "left" else width - 1
        field[start : end + 1, column] = value


def _spatial_seed_ownership(
    shape,
    confirmed_contacts,
    skipped_candidates,
    component_id,
    blend_width,
):
    """Assign a component locally to confirmed or skipped edge seeds in 2D."""
    relevant_skips = [
        item
        for item in skipped_candidates
        if item.get("component_id") == int(component_id)
    ]
    if not relevant_skips:
        return None

    confirmed_distance = np.full(shape, np.inf, dtype=np.float32)
    for contact in confirmed_contacts:
        confirmed_distance = np.minimum(
            confirmed_distance,
            _edge_seed_distance(
                shape,
                contact["side"],
                contact["start"],
                contact["end"],
            ),
        )
    skipped_distance = np.full(shape, np.inf, dtype=np.float32)
    for skipped in relevant_skips:
        skipped_distance = np.minimum(
            skipped_distance,
            _edge_seed_distance(
                shape,
                skipped["side"],
                skipped["start"],
                skipped["end"],
            ),
        )

    blend_width = max(1.0, float(blend_width))
    ownership = _smootherstep(
        0.5
        + 0.5
        * (skipped_distance - confirmed_distance)
        / blend_width
    ).astype(np.float32)
    for contact in confirmed_contacts:
        _write_edge_seed(
            ownership,
            contact["side"],
            contact["start"],
            contact["end"],
            1.0,
        )
    # Ambiguous shared corner pixels are precision-first: a skipped seed wins.
    for skipped in relevant_skips:
        _write_edge_seed(
            ownership,
            skipped["side"],
            skipped["start"],
            skipped["end"],
            0.0,
        )
    return ownership


def _bowed_weight(lateral, start, end, padding, end_depth_ratio=0.42):
    center = 0.5 * (float(start) + float(end))
    half = max(0.5, 0.5 * float(end - start))
    distance = np.abs(lateral - center)
    weight = np.zeros_like(lateral, dtype=np.float32)

    inside = distance <= half
    normalized_inside = distance[inside] / half
    weight[inside] = end_depth_ratio + (1.0 - end_depth_ratio) * (
        1.0 - _smootherstep(normalized_inside)
    )

    if padding > 0:
        padded = (distance > half) & (distance < half + padding)
        normalized_padding = (distance[padded] - half) / float(padding)
        weight[padded] = end_depth_ratio * (
            1.0 - _smootherstep(normalized_padding)
        )
    return weight


def _hard_inset_weight(lateral, start, end, padding):
    weight = np.zeros_like(lateral, dtype=np.float32)
    inside = (lateral >= float(start)) & (lateral <= float(end))
    weight[inside] = 1.0
    if padding > 0:
        before = (lateral < float(start)) & (
            lateral > float(start) - float(padding)
        )
        after = (lateral > float(end)) & (
            lateral < float(end) + float(padding)
        )
        weight[before] = 1.0 - _smootherstep(
            (float(start) - lateral[before]) / float(padding)
        )
        weight[after] = 1.0 - _smootherstep(
            (lateral[after] - float(end)) / float(padding)
        )
    return weight


def _seeded_8_connected_mask(visible, edge_probe, seed_start, seed_end):
    """Return all low-alpha runs 8-connected to a confirmed outer-edge seed."""
    visible = np.asarray(visible, dtype=bool)
    height, width = visible.shape
    parent = []
    rank = []
    runs = []
    seeded_ids = set()

    def make_set():
        identifier = len(parent)
        parent.append(identifier)
        rank.append(0)
        return identifier

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    previous = []
    probe_limit = min(height, max(1, int(edge_probe)))
    seed_start = max(0, min(width - 1, int(seed_start)))
    seed_end = max(seed_start, min(width - 1, int(seed_end)))
    for y in range(height):
        padded = np.pad(visible[y].astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        current = []
        previous_index = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            identifier = make_set()
            while (
                previous_index < len(previous)
                and previous[previous_index][1] < start - 1
            ):
                previous_index += 1
            overlap_index = previous_index
            while (
                overlap_index < len(previous)
                and previous[overlap_index][0] <= end + 1
            ):
                union(identifier, previous[overlap_index][2])
                overlap_index += 1
            if y < probe_limit and end >= seed_start and start <= seed_end:
                seeded_ids.add(identifier)
            current.append((int(start), int(end), identifier))
            runs.append((y, int(start), int(end), identifier))
        previous = current

    mask = np.zeros((height, width), dtype=np.float32)
    seeded_roots = {find(identifier) for identifier in seeded_ids}
    for y, start, end, identifier in runs:
        if find(identifier) in seeded_roots:
            mask[y, start : end + 1] = 1.0
    return mask


def _oriented_edge_alpha(alpha, side, lower, upper, max_depth):
    """Return a patch as (inward distance, lateral coordinate)."""
    height, width = alpha.shape
    if side == "top":
        return alpha[:max_depth, lower : upper + 1]
    if side == "bottom":
        return alpha[height - max_depth :, lower : upper + 1][::-1, :]
    if side == "left":
        return alpha[lower : upper + 1, :max_depth].T
    return alpha[lower : upper + 1, width - max_depth :][:, ::-1].T


def _component_in_edge_layout(component, side):
    """Map an inward/lateral component mask back to its guard patch layout."""
    if side == "top":
        return component
    if side == "bottom":
        return component[::-1, :]
    if side == "left":
        return component.T
    return component.T[:, ::-1]


def _edge_contact_component(
    alpha,
    side,
    lower,
    upper,
    max_depth,
    start,
    end,
    edge_probe,
    outer_threshold,
    component_labels=None,
    component_id=None,
):
    oriented_alpha = _oriented_edge_alpha(
        alpha,
        side,
        lower,
        upper,
        max_depth,
    )
    if component_labels is not None and component_id is not None:
        oriented_labels = _oriented_edge_alpha(
            component_labels,
            side,
            lower,
            upper,
            max_depth,
        )
        component = (oriented_labels == int(component_id)).astype(np.float32)
    else:
        component = _seeded_8_connected_mask(
            oriented_alpha >= float(outer_threshold),
            1,
            int(start) - lower,
            int(end) - lower,
        )
    return component, _component_in_edge_layout(component, side)


def _multiply_bowed_edge_guard(
    guard,
    alpha,
    side,
    start,
    end,
    hard_zero,
    feather,
    lateral_padding,
    edge_probe,
    outer_threshold,
    component_labels=None,
    component_id=None,
    lateral_ownership=None,
    spatial_ownership=None,
):
    height, width = guard.shape
    side_length = width if side in ("top", "bottom") else height
    lower = max(0, int(start) - lateral_padding)
    upper = min(side_length - 1, int(end) + lateral_padding)
    lateral = np.arange(lower, upper + 1, dtype=np.float32)
    feather_weight = _bowed_weight(lateral, start, end, lateral_padding)
    hard_weight = _hard_inset_weight(lateral, start, end, lateral_padding)
    max_depth = max(1, min(
        height if side in ("top", "bottom") else width,
        int(np.ceil(hard_zero + feather)) + 2,
    ))

    if side == "top":
        inward = np.arange(max_depth, dtype=np.float32)[:, None]
        feather_scale = feather_weight[None, :]
        hard_scale = hard_weight[None, :]
    elif side == "bottom":
        inward = np.arange(max_depth - 1, -1, -1, dtype=np.float32)[:, None]
        feather_scale = feather_weight[None, :]
        hard_scale = hard_weight[None, :]
    elif side == "left":
        inward = np.arange(max_depth, dtype=np.float32)[None, :]
        feather_scale = feather_weight[:, None]
        hard_scale = hard_weight[:, None]
    else:
        inward = np.arange(max_depth - 1, -1, -1, dtype=np.float32)[None, :]
        feather_scale = feather_weight[:, None]
        hard_scale = hard_weight[:, None]

    local_shape = np.broadcast_shapes(inward.shape, feather_scale.shape)
    local = np.ones(local_shape, dtype=np.float32)
    feather_scale = np.broadcast_to(feather_scale, local_shape)
    hard_scale = np.broadcast_to(hard_scale, local_shape)
    inward = np.broadcast_to(inward, local_shape)
    active = feather_scale > 1.0e-6
    denominator = np.maximum(float(feather) * feather_scale, 1.0e-6)
    transition = (inward - float(hard_zero) * hard_scale) / denominator
    local[active] = _smootherstep(transition)[active]
    component, component_2d = _edge_contact_component(
        alpha,
        side,
        lower,
        upper,
        max_depth,
        start,
        end,
        edge_probe,
        outer_threshold,
        component_labels,
        component_id,
    )
    component_2d = np.broadcast_to(component_2d, local_shape)
    if lateral_ownership is not None:
        ownership = np.asarray(
            lateral_ownership[lower : upper + 1], dtype=np.float32
        )
        if side in ("top", "bottom"):
            ownership = ownership[None, :]
        else:
            ownership = ownership[:, None]
        component_2d = component_2d * np.broadcast_to(ownership, local_shape)
    if spatial_ownership is not None:
        if side == "top":
            ownership_2d = spatial_ownership[:max_depth, lower : upper + 1]
        elif side == "bottom":
            ownership_2d = spatial_ownership[
                height - max_depth :, lower : upper + 1
            ]
        elif side == "left":
            ownership_2d = spatial_ownership[lower : upper + 1, :max_depth]
        else:
            ownership_2d = spatial_ownership[
                lower : upper + 1, width - max_depth :
            ]
        component_2d = component_2d * ownership_2d
    local = 1.0 - component_2d * (1.0 - local)

    if side == "top":
        guard[:max_depth, lower : upper + 1] *= local
    elif side == "bottom":
        guard[height - max_depth :, lower : upper + 1] *= local
    elif side == "left":
        guard[lower : upper + 1, :max_depth] *= local
    else:
        guard[lower : upper + 1, width - max_depth :] *= local
    return int(np.count_nonzero(component))


def _wide_circle_geometry(side_length, normal_length, hard_zero, feather):
    """Return a gentle virtual circle whose edge arc bends toward canvas center."""
    lateral_radius = max(0.5, 0.5 * float(side_length - 1))
    target_sagitta = min(
        max(1.0, 0.45 * float(feather)),
        max(1.0, 0.18 * float(normal_length)),
    )
    radius = (
        lateral_radius * lateral_radius + target_sagitta * target_sagitta
    ) / (2.0 * target_sagitta)
    # Every point in the local support must be able to reach guard=1 on the
    # near side of the circle, including at the largest allowed feather.
    radius = max(
        radius,
        lateral_radius + float(hard_zero) + float(feather) + 1.0,
    )
    actual_sagitta = radius - np.sqrt(
        max(radius * radius - lateral_radius * lateral_radius, 0.0)
    )
    return lateral_radius, float(radius), float(actual_sagitta)


def _multiply_wide_circle_arc_guard(
    guard,
    alpha,
    side,
    start,
    end,
    hard_zero,
    feather,
    lateral_padding,
    edge_probe,
    outer_threshold,
    component_labels=None,
    component_id=None,
    lateral_ownership=None,
    spatial_ownership=None,
):
    """Multiply one locally gated true-circle arc for a confirmed wide contact."""
    height, width = guard.shape
    side_length = width if side in ("top", "bottom") else height
    normal_length = height if side in ("top", "bottom") else width
    lower = max(0, int(start) - lateral_padding)
    upper = min(side_length - 1, int(end) + lateral_padding)
    lateral = np.arange(lower, upper + 1, dtype=np.float32)
    support = _hard_inset_weight(lateral, start, end, lateral_padding)

    requested_hard_zero = float(hard_zero)
    requested_feather = float(feather)
    center = max(0.5, 0.5 * float(side_length - 1))
    lateral_offset = lateral - float(center)
    max_offset = float(np.max(np.abs(lateral_offset[support > 1.0e-6])))

    def resolve_geometry(parameter_scale):
        effective_hard_zero = requested_hard_zero * float(parameter_scale)
        effective_feather = max(
            1.0,
            requested_feather * float(parameter_scale),
        )
        _, local_radius, local_sagitta = _wide_circle_geometry(
            side_length,
            normal_length,
            effective_hard_zero,
            effective_feather,
        )
        inner_radius = (
            local_radius - effective_hard_zero - effective_feather
        )
        radicand = max(
            inner_radius * inner_radius - max_offset * max_offset,
            0.0,
        )
        completion_depth = local_radius - np.sqrt(radicand)
        return (
            effective_hard_zero,
            effective_feather,
            local_radius,
            local_sagitta,
            float(completion_depth),
        )

    parameter_scale = 1.0
    (
        effective_hard_zero,
        effective_feather,
        radius,
        actual_sagitta,
        completion_depth,
    ) = resolve_geometry(parameter_scale)
    completion_limit = max(1.0, float(normal_length - 2))
    if completion_depth > completion_limit and normal_length > 2:
        lower_scale = 0.0
        upper_scale = 1.0
        for _ in range(32):
            candidate_scale = 0.5 * (lower_scale + upper_scale)
            candidate = resolve_geometry(candidate_scale)
            if candidate[-1] <= completion_limit:
                lower_scale = candidate_scale
            else:
                upper_scale = candidate_scale
        parameter_scale = lower_scale
        (
            effective_hard_zero,
            effective_feather,
            radius,
            actual_sagitta,
            completion_depth,
        ) = resolve_geometry(parameter_scale)

    max_depth = max(
        1,
        min(
            normal_length,
            int(np.ceil(completion_depth)) + 2,
        ),
    )

    component, component_2d = _edge_contact_component(
        alpha,
        side,
        lower,
        upper,
        max_depth,
        start,
        end,
        edge_probe,
        outer_threshold,
        component_labels,
        component_id,
    )

    if side == "top":
        inward = np.arange(max_depth, dtype=np.float32)[:, None]
        offset = lateral_offset[None, :]
        support_2d = support[None, :]
    elif side == "bottom":
        inward = np.arange(max_depth - 1, -1, -1, dtype=np.float32)[:, None]
        offset = lateral_offset[None, :]
        support_2d = support[None, :]
    elif side == "left":
        inward = np.arange(max_depth, dtype=np.float32)[None, :]
        offset = lateral_offset[:, None]
        support_2d = support[:, None]
    else:
        inward = np.arange(max_depth - 1, -1, -1, dtype=np.float32)[None, :]
        offset = lateral_offset[:, None]
        support_2d = support[:, None]

    local_shape = np.broadcast_shapes(inward.shape, offset.shape)
    inward = np.broadcast_to(inward, local_shape)
    offset = np.broadcast_to(offset, local_shape)
    support_2d = np.broadcast_to(support_2d, local_shape)
    component_2d = np.broadcast_to(component_2d, local_shape)
    if lateral_ownership is not None:
        ownership = np.asarray(
            lateral_ownership[lower : upper + 1], dtype=np.float32
        )
        if side in ("top", "bottom"):
            ownership = ownership[None, :]
        else:
            ownership = ownership[:, None]
        component_2d = component_2d * np.broadcast_to(ownership, local_shape)
    if spatial_ownership is not None:
        if side == "top":
            ownership_2d = spatial_ownership[:max_depth, lower : upper + 1]
        elif side == "bottom":
            ownership_2d = spatial_ownership[
                height - max_depth :, lower : upper + 1
            ]
        elif side == "left":
            ownership_2d = spatial_ownership[lower : upper + 1, :max_depth]
        else:
            ownership_2d = spatial_ownership[
                lower : upper + 1, width - max_depth :
            ]
        component_2d = component_2d * ownership_2d
    radial_distance = np.sqrt(
        offset * offset + (float(radius) - inward) ** 2
    )
    signed_inside = float(radius) - radial_distance
    circle = _smootherstep(
        (signed_inside - effective_hard_zero) / effective_feather
    )
    local = 1.0 - support_2d * component_2d * (1.0 - circle)

    contact_offset = max(
        abs(float(start) - float(center)),
        abs(float(end) - float(center)),
    )
    contact_sagitta = float(radius) - np.sqrt(
        max(float(radius) ** 2 - contact_offset**2, 0.0)
    )
    support_sagitta = float(radius) - np.sqrt(
        max(float(radius) ** 2 - max_offset**2, 0.0)
    )

    if side == "top":
        guard[:max_depth, lower : upper + 1] *= local
    elif side == "bottom":
        guard[height - max_depth :, lower : upper + 1] *= local
    elif side == "left":
        guard[lower : upper + 1, :max_depth] *= local
    else:
        guard[lower : upper + 1, width - max_depth :] *= local

    return {
        "arc_lateral_center_px": round(float(center), 3),
        "arc_radius_px": round(float(radius), 3),
        "arc_canvas_sagitta_px": round(float(actual_sagitta), 3),
        "arc_contact_sagitta_px": round(float(contact_sagitta), 3),
        "arc_support_sagitta_px": round(float(support_sagitta), 3),
        "arc_max_depth_px": int(max_depth),
        "arc_component_pixels": int(np.count_nonzero(component)),
        "arc_parameter_scale": round(float(parameter_scale), 6),
        "arc_effective_hard_zero_px": round(float(effective_hard_zero), 3),
        "arc_effective_feather_px": round(float(effective_feather), 3),
    }


def _corner_gap(side, segment, corner, height, width):
    start, end = segment["start"], segment["end"]
    if side in ("top", "bottom"):
        return start if corner.endswith("left") else width - 1 - end
    return start if corner.startswith("top") else height - 1 - end


def _corner_extent(side, segment, corner, height, width):
    start, end = segment["start"], segment["end"]
    if side in ("top", "bottom"):
        return end if corner.endswith("left") else width - 1 - start
    return end if corner.startswith("top") else height - 1 - start


def _corner_ellipse_guard(axis_x, axis_y, hard_zero):
    x = np.arange(axis_x, dtype=np.float32)[None, :]
    y = np.arange(axis_y, dtype=np.float32)[:, None]
    denominator_x = max(float(axis_x - 1 - hard_zero), 1.0)
    denominator_y = max(float(axis_y - 1 - hard_zero), 1.0)
    u = np.maximum(x - float(hard_zero), 0.0) / denominator_x
    v = np.maximum(y - float(hard_zero), 0.0) / denominator_y
    inside = u * u + v * v < 1.0
    local = np.ones((axis_y, axis_x), dtype=np.float32)
    denominator = np.sqrt(
        np.maximum((1.0 - u * u) * (1.0 - v * v), 1.0e-12)
    )
    transition = np.broadcast_to((u * v) / denominator, local.shape)
    local[inside] = _smootherstep(transition[inside])
    return local


def _multiply_corner_guard(
    guard,
    alpha,
    corner,
    horizontal_side,
    horizontal,
    vertical_side,
    vertical,
    axis_x,
    axis_y,
    hard_zero,
    edge_probe,
    outer_threshold,
    component_labels=None,
    component_id=None,
    horizontal_ownership=None,
    vertical_ownership=None,
    spatial_ownership=None,
):
    height, width = guard.shape
    axis_x = max(1, min(width, int(axis_x)))
    axis_y = max(1, min(height, int(axis_y)))
    local = _corner_ellipse_guard(axis_x, axis_y, hard_zero)
    visible = _corner_alpha_patch(alpha, corner, axis_x, axis_y) >= float(
        outer_threshold
    )
    horizontal_start, horizontal_end = _corner_run_coordinates(
        horizontal_side,
        horizontal,
        corner,
        *alpha.shape,
    )
    vertical_start, vertical_end = _corner_run_coordinates(
        vertical_side,
        vertical,
        corner,
        *alpha.shape,
    )
    if component_labels is not None and component_id is not None:
        component = (
            _corner_alpha_patch(component_labels, corner, axis_x, axis_y)
            == int(component_id)
        ).astype(np.float32)
    else:
        horizontal_component = _seeded_8_connected_mask(
            visible,
            1,
            horizontal_start,
            horizontal_end,
        )
        vertical_component = _seeded_8_connected_mask(
            visible.T,
            1,
            vertical_start,
            vertical_end,
        ).T
        component = np.maximum(horizontal_component, vertical_component)
    if horizontal_ownership is not None and vertical_ownership is not None:
        horizontal_ownership = np.asarray(
            horizontal_ownership, dtype=np.float32
        )
        vertical_ownership = np.asarray(vertical_ownership, dtype=np.float32)
        if corner.endswith("left"):
            horizontal_local = horizontal_ownership[:axis_x]
        else:
            horizontal_local = horizontal_ownership[width - axis_x :][::-1]
        if corner.startswith("top"):
            vertical_local = vertical_ownership[:axis_y]
        else:
            vertical_local = vertical_ownership[height - axis_y :][::-1]
        ownership = np.minimum(
            horizontal_local[None, :],
            vertical_local[:, None],
        )
        component *= ownership
    if spatial_ownership is not None:
        component *= _corner_alpha_patch(
            spatial_ownership,
            corner,
            axis_x,
            axis_y,
        )
    local = 1.0 - component * (1.0 - local)
    if corner == "top_left":
        guard[:axis_y, :axis_x] *= local
    elif corner == "top_right":
        guard[:axis_y, width - axis_x :] *= local[:, ::-1]
    elif corner == "bottom_left":
        guard[height - axis_y :, :axis_x] *= local[::-1, :]
    else:
        guard[height - axis_y :, width - axis_x :] *= local[::-1, ::-1]
    return int(np.count_nonzero(component))


def _corner_alpha_patch(alpha, corner, axis_x, axis_y):
    height, width = alpha.shape
    axis_x = max(1, min(width, int(axis_x)))
    axis_y = max(1, min(height, int(axis_y)))
    if corner == "top_left":
        return alpha[:axis_y, :axis_x]
    if corner == "top_right":
        return alpha[:axis_y, width - axis_x :][:, ::-1]
    if corner == "bottom_left":
        return alpha[height - axis_y :, :axis_x][::-1, :]
    return alpha[height - axis_y :, width - axis_x :][::-1, ::-1]


def _corner_run_coordinates(side, segment, corner, height, width):
    start, end = int(segment["start"]), int(segment["end"])
    if side in ("top", "bottom"):
        if corner.endswith("left"):
            return start, end
        return width - 1 - end, width - 1 - start
    if corner.startswith("top"):
        return start, end
    return height - 1 - end, height - 1 - start


def _corner_segments_connected(
    alpha,
    corner,
    horizontal_side,
    horizontal,
    vertical_side,
    vertical,
    axis_x,
    axis_y,
    edge_probe,
    outer_threshold,
    connectivity_threshold,
):
    """Check 8-connected Alpha before replacing two local bows with one corner field."""
    patch = _corner_alpha_patch(alpha, corner, axis_x, axis_y)
    outer_visible = patch >= float(outer_threshold)
    visible = patch >= float(connectivity_threshold)
    if not np.any(outer_visible):
        return False

    horizontal_start, horizontal_end = _corner_run_coordinates(
        horizontal_side, horizontal, corner, *alpha.shape
    )
    vertical_start, vertical_end = _corner_run_coordinates(
        vertical_side, vertical, corner, *alpha.shape
    )
    horizontal_start = max(0, min(patch.shape[1] - 1, horizontal_start))
    horizontal_end = max(0, min(patch.shape[1] - 1, horizontal_end))
    vertical_start = max(0, min(patch.shape[0] - 1, vertical_start))
    vertical_end = max(0, min(patch.shape[0] - 1, vertical_end))
    probe_y = 1
    probe_x = 1

    source = np.zeros_like(visible, dtype=bool)
    source[:probe_y, horizontal_start : horizontal_end + 1] = True
    source &= outer_visible
    target = np.zeros_like(visible, dtype=bool)
    target[vertical_start : vertical_end + 1, :probe_x] = True
    target &= outer_visible
    if not np.any(source) or not np.any(target):
        return False

    traversable = visible | source | target
    visited = source.copy()
    queue = deque(map(tuple, np.argwhere(source)))
    patch_height, patch_width = visible.shape
    while queue:
        y, x = queue.popleft()
        if target[y, x]:
            return True
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                if delta_y == 0 and delta_x == 0:
                    continue
                next_y = y + delta_y
                next_x = x + delta_x
                if (
                    0 <= next_y < patch_height
                    and 0 <= next_x < patch_width
                    and traversable[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
    return False


def build_adaptive_edge_guard(
    alpha,
    safety_inset=8,
    feather_width=240,
    contact_threshold=0.03,
    return_skipped=False,
):
    """Build local curved guards only where crop-like edge evidence is strong."""
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim != 2:
        raise ValueError("alpha 必须是二维灰度数组。")

    height, width = alpha.shape
    # Parameters are expressed in the canonical 1280 output space. Contain-resize
    # maps the source's longest side to 1280, so scale against that same side.
    scale = max(height, width) / float(REFERENCE_SIZE)
    hard_zero = max(0, int(round(float(safety_inset) * scale)))
    base_feather = max(1, int(round(float(feather_width) * scale)))
    edge_probe = max(1, int(round(4 * scale)))
    scan_band = max(1, int(round(48 * scale)))
    merge_gap = max(1, int(round(12 * scale)))
    run_extension = max(1, int(round(24 * scale)))
    lateral_padding = max(2, int(round(64 * scale)))
    corner_snap = max(2, int(round(128 * scale)))
    corner_length_limit = max(2, int(round(2.0 * base_feather)))
    guard = np.ones((height, width), dtype=np.float32)
    contacts_by_side = {side: [] for side in ("top", "right", "bottom", "left")}
    candidate_segments_by_side = {
        side: [] for side in ("top", "right", "bottom", "left")
    }
    skipped_candidates = []
    span_threshold = float(contact_threshold)
    outer_threshold = min(span_threshold, 0.5 / 255.0)
    connectivity_threshold = min(span_threshold, 1.0 / 255.0)
    labels = _label_8_connected_components(alpha >= outer_threshold)

    for side in ("top", "right", "bottom", "left"):
        candidate_segments_by_side[side] = _edge_candidates(
            alpha,
            labels,
            side,
            edge_probe,
            scan_band,
            outer_threshold,
            span_threshold,
            run_extension,
        )

    balanced_frames = _balanced_frame_flush_evidence(
        alpha,
        labels,
        candidate_segments_by_side,
        edge_probe,
        span_threshold,
    )

    for side in ("top", "right", "bottom", "left"):
        side_length = width if side in ("top", "bottom") else height
        for candidate in candidate_segments_by_side[side]:
            start = int(candidate["seed_start"])
            end = int(candidate["seed_end"])
            component_id = int(candidate["component_id"])
            candidate_diagnostics = {
                "side": side,
                "component_id": component_id,
                "start": start,
                "end": end,
                "length": int(end - start + 1),
                "span_start": int(candidate["span_start"]),
                "span_end": int(candidate["span_end"]),
            }
            if component_id in balanced_frames:
                skipped_candidates.append(
                    {
                        **candidate_diagnostics,
                        **balanced_frames[component_id],
                    }
                )
                continue
            confirmed, evidence = _edge_segment_clip_evidence(
                alpha,
                labels,
                candidate,
                edge_probe,
                scan_band,
                outer_threshold,
                span_threshold,
                scale,
            )
            if not confirmed:
                skipped_candidates.append(
                    {
                        **candidate_diagnostics,
                        **evidence,
                    }
                )
                continue
            length = end - start + 1
            fraction = length / float(side_length)
            is_wide = fraction >= 0.45
            local_feather = max(
                1,
                int(round(base_feather * _wide_multiplier(fraction))),
            )
            contacts_by_side[side].append(
                {
                    "side": side,
                    "component_id": component_id,
                    "start": int(start),
                    "end": int(end),
                    "span_start": int(candidate["span_start"]),
                    "span_end": int(candidate["span_end"]),
                    "length": int(length),
                    "fraction": round(fraction, 6),
                    "wide": bool(is_wide),
                    "hard_zero_px": int(hard_zero),
                    "feather_px": int(local_feather),
                    "edge_probe_px": int(edge_probe),
                    "run_extension_px": int(run_extension),
                    "lateral_padding_px": int(lateral_padding),
                    "outer_threshold": float(outer_threshold),
                    "span_threshold": float(span_threshold),
                    "connectivity_threshold": float(connectivity_threshold),
                    "support_start": int(max(0, start - lateral_padding)),
                    "support_end": int(min(side_length - 1, end + lateral_padding)),
                    "mode": "bowed_edge",
                    "consumed": False,
                    **evidence,
                }
            )

    for side in ("top", "right", "bottom", "left"):
        side_length = width if side in ("top", "bottom") else height
        side_skips = [
            item for item in skipped_candidates if item["side"] == side
        ]
        merged_contacts = []
        for contact in sorted(contacts_by_side[side], key=lambda item: item["start"]):
            if not merged_contacts:
                merged_contacts.append(contact)
                continue
            previous = merged_contacts[-1]
            union_start = previous["start"]
            union_end = contact["end"]
            gap = contact["start"] - previous["end"] - 1
            blocked_by_skip = any(
                item.get("component_id") == contact["component_id"]
                and item["end"] >= union_start
                and item["start"] <= union_end
                for item in side_skips
            )
            if (
                previous["component_id"] != contact["component_id"]
                or gap > merge_gap
                or blocked_by_skip
            ):
                merged_contacts.append(contact)
                continue

            source_segments = previous.pop(
                "source_segments",
                [[previous["start"], previous["end"]]],
            )
            source_segments.extend(
                contact.get(
                    "source_segments",
                    [[contact["start"], contact["end"]]],
                )
            )
            previous["source_segments"] = source_segments
            previous["end"] = int(contact["end"])
            previous["span_start"] = min(
                previous["span_start"], contact["span_start"]
            )
            previous["span_end"] = max(
                previous["span_end"], contact["span_end"]
            )
            previous["length"] = previous["end"] - previous["start"] + 1
            fraction = previous["length"] / float(side_length)
            previous["fraction"] = round(fraction, 6)
            previous["wide"] = bool(fraction >= 0.45)
            previous["feather_px"] = max(
                1,
                int(round(base_feather * _wide_multiplier(fraction))),
            )
            previous["support_start"] = int(
                max(0, previous["start"] - lateral_padding)
            )
            previous["support_end"] = int(
                min(side_length - 1, previous["end"] + lateral_padding)
            )
            previous["actual_border_pixels"] += contact[
                "actual_border_pixels"
            ]
            previous["robust_border_run_px"] += contact[
                "robust_border_run_px"
            ]
            previous["border_fill_p25"] = min(
                previous["border_fill_p25"], contact["border_fill_p25"]
            )
            previous["border_robust_fraction"] = min(
                previous["border_robust_fraction"],
                contact["border_robust_fraction"],
            )
            previous["inward_width_p90_px"] = max(
                previous["inward_width_p90_px"],
                contact["inward_width_p90_px"],
            )
            previous["width_persistence"] = min(
                previous["width_persistence"], contact["width_persistence"]
            )
        contacts_by_side[side] = merged_contacts

    ownership_blend = max(1, int(round(12 * scale)))
    for side in ("top", "right", "bottom", "left"):
        side_length = width if side in ("top", "bottom") else height
        for contact in contacts_by_side[side]:
            skipped_ranges = [
                (item["start"], item["end"])
                for item in skipped_candidates
                if item["side"] == side
                and item.get("component_id") == contact["component_id"]
            ]
            contact["_lateral_ownership"] = _lateral_ownership_profile(
                side_length,
                contact["start"],
                contact["end"],
                skipped_ranges,
                ownership_blend,
            )

    corner_pairs = {
        "top_left": ("top", "left"),
        "top_right": ("top", "right"),
        "bottom_right": ("bottom", "right"),
        "bottom_left": ("bottom", "left"),
    }
    valid_corner_candidates = []
    for corner, (horizontal_side, vertical_side) in corner_pairs.items():
        for horizontal in contacts_by_side[horizontal_side]:
            if horizontal["length"] > corner_length_limit:
                continue
            horizontal_gap = _corner_gap(
                horizontal_side, horizontal, corner, height, width
            )
            if horizontal_gap > corner_snap:
                continue
            for vertical in contacts_by_side[vertical_side]:
                if horizontal["component_id"] != vertical["component_id"]:
                    continue
                if vertical["length"] > corner_length_limit:
                    continue
                vertical_gap = _corner_gap(
                    vertical_side, vertical, corner, height, width
                )
                if vertical_gap > corner_snap:
                    continue
                corner_feather = max(
                    horizontal["feather_px"], vertical["feather_px"]
                )
                axis_x = max(
                    hard_zero + corner_feather + 1,
                    _corner_extent(
                        horizontal_side, horizontal, corner, height, width
                    )
                    + lateral_padding
                    + 1,
                )
                axis_y = max(
                    hard_zero + corner_feather + 1,
                    _corner_extent(
                        vertical_side, vertical, corner, height, width
                    )
                    + lateral_padding
                    + 1,
                )
                axis_x = min(width, int(axis_x))
                axis_y = min(height, int(axis_y))
                if not _corner_segments_connected(
                    alpha,
                    corner,
                    horizontal_side,
                    horizontal,
                    vertical_side,
                    vertical,
                    axis_x,
                    axis_y,
                    edge_probe,
                    outer_threshold,
                    connectivity_threshold,
                ):
                    continue
                valid_corner_candidates.append(
                    {
                        "corner": corner,
                        "horizontal_side": horizontal_side,
                        "vertical_side": vertical_side,
                        "horizontal": horizontal,
                        "vertical": vertical,
                        "axis_x": axis_x,
                        "axis_y": axis_y,
                    }
                )

    # A contact that can pair with more than one corner has no rotation-stable
    # winner.  Keep every such candidate as independent edge fields instead of
    # letting the fixed corner iteration order choose an arbitrary ellipse.
    corner_participation = {}
    for candidate in valid_corner_candidates:
        for contact in (candidate["horizontal"], candidate["vertical"]):
            identifier = id(contact)
            corner_participation[identifier] = (
                corner_participation.get(identifier, 0) + 1
            )
    for candidate in valid_corner_candidates:
        horizontal = candidate["horizontal"]
        vertical = candidate["vertical"]
        if (
            corner_participation[id(horizontal)] != 1
            or corner_participation[id(vertical)] != 1
        ):
            horizontal["corner_merge_skipped"] = "ambiguous_multi_corner"
            vertical["corner_merge_skipped"] = "ambiguous_multi_corner"
            continue

        corner = candidate["corner"]
        horizontal_side = candidate["horizontal_side"]
        vertical_side = candidate["vertical_side"]
        axis_x = candidate["axis_x"]
        axis_y = candidate["axis_y"]
        corner_spatial_ownership = _spatial_seed_ownership(
            alpha.shape,
            [horizontal, vertical],
            skipped_candidates,
            horizontal["component_id"],
            ownership_blend,
        )
        component_pixels = _multiply_corner_guard(
            guard,
            alpha,
            corner,
            horizontal_side,
            horizontal,
            vertical_side,
            vertical,
            axis_x,
            axis_y,
            hard_zero,
            edge_probe,
            outer_threshold,
            labels,
            horizontal["component_id"],
            horizontal["_lateral_ownership"],
            vertical["_lateral_ownership"],
            corner_spatial_ownership,
        )
        for contact in (horizontal, vertical):
            contact["mode"] = "corner_ellipse"
            contact["corner"] = corner
            contact["corner_axis_px"] = [int(axis_x), int(axis_y)]
            contact["component_pixels"] = component_pixels
            contact["consumed"] = True

    contacts = []
    for side in ("top", "right", "bottom", "left"):
        for contact in contacts_by_side[side]:
            if not contact["consumed"]:
                spatial_ownership = _spatial_seed_ownership(
                    alpha.shape,
                    [contact],
                    skipped_candidates,
                    contact["component_id"],
                    ownership_blend,
                )
                if contact["wide"]:
                    arc_diagnostics = _multiply_wide_circle_arc_guard(
                        guard,
                        alpha,
                        side,
                        contact["start"],
                        contact["end"],
                        hard_zero,
                        contact["feather_px"],
                        lateral_padding,
                        edge_probe,
                        outer_threshold,
                        labels,
                        contact["component_id"],
                        contact["_lateral_ownership"],
                        spatial_ownership,
                    )
                    contact["mode"] = "wide_circle_arc"
                    contact.update(arc_diagnostics)
                else:
                    contact["component_pixels"] = _multiply_bowed_edge_guard(
                        guard,
                        alpha,
                        side,
                        contact["start"],
                        contact["end"],
                        hard_zero,
                        contact["feather_px"],
                        lateral_padding,
                        edge_probe,
                        outer_threshold,
                        labels,
                        contact["component_id"],
                        contact["_lateral_ownership"],
                        spatial_ownership,
                    )
            contact.pop("_lateral_ownership", None)
            contact.pop("consumed", None)
            contacts.append(contact)

    result = (np.clip(guard, 0.0, 1.0).astype(np.float32), contacts)
    if return_skipped:
        return (*result, skipped_candidates)
    return result


def apply_adaptive_edge_fade(
    rgb,
    alpha,
    safety_inset=8,
    feather_width=240,
    contact_threshold=0.03,
    return_skipped=False,
):
    rgb = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb 必须是 HWC RGB/RGBA 数组。")
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("final_alpha 与 rgba 的高宽必须一致。")

    rgb = np.clip(rgb[..., :3], 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    guard_result = build_adaptive_edge_guard(
        alpha,
        safety_inset=safety_inset,
        feather_width=feather_width,
        contact_threshold=contact_threshold,
        return_skipped=return_skipped,
    )
    if return_skipped:
        guard, contacts, skipped_candidates = guard_result
    else:
        guard, contacts = guard_result
    output_alpha = alpha * guard
    result = (rgb, output_alpha.astype(np.float32), guard, contacts)
    if return_skipped:
        return (*result, skipped_candidates)
    return result


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
                    {"default": 8, "min": 0, "max": 256, "step": 1},
                ),
                "feather_width": (
                    "INT",
                    {"default": 240, "min": 8, "max": 640, "step": 1},
                ),
                "contact_threshold": (
                    "FLOAT",
                    {
                        "default": 0.03,
                        "min": 0.001,
                        "max": 0.5,
                        "step": 0.001,
                        "tooltip": "用于估计候选主体范围；最外低阈值只负责找候选，实际第 0 行／列还须通过连续实心宽度、约 12px 向内支撑与宽度持久性证据。贴边、相切、浅层或稀疏边界保持不变。",
                    },
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
        safety_inset=8,
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
            (
                rgb,
                repaired_alpha,
                guard,
                contacts,
                skipped_candidates,
            ) = apply_adaptive_edge_fade(
                image[..., :3],
                alpha,
                safety_inset=safety_inset,
                feather_width=feather_width,
                contact_threshold=contact_threshold,
                return_skipped=True,
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
                    "diagnostics_schema": "clip_evidence_v4",
                    "input_size": [int(image.shape[1]), int(image.shape[0])],
                    "contacts": contacts,
                    "skipped_candidates": skipped_candidates,
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
