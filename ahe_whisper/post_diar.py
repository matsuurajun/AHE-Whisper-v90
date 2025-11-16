# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger("ahe_whisper_post_diar")


def collect_speaker_stats(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except (TypeError, ValueError):
            continue

        dur = end - start
        if dur <= 0.0:
            continue

        rec = stats.setdefault(spk, {"duration": 0.0, "count": 0.0})
        rec["duration"] += dur
        rec["count"] += 1.0

    return stats


def infer_effective_speakers(
    stats: Dict[str, Dict[str, float]],
    coverage_th: float = 0.97,
    max_speakers: int | None = None,
) -> Tuple[List[str], List[str]]:
    if not stats:
        return [], []

    items = sorted(stats.items(), key=lambda kv: kv[1]["duration"], reverse=True)
    total = sum(v["duration"] for _, v in items) or 1.0

    keep: List[str] = []
    acc = 0.0
    for spk, rec in items:
        keep.append(spk)
        acc += rec["duration"]
        if acc / total >= coverage_th:
            break

    if max_speakers is not None and max_speakers > 0 and len(keep) > max_speakers:
        keep = keep[:max_speakers]

    drop = [spk for spk, _ in items if spk not in keep]
    return keep, drop


def _nearest_keep_speaker(
    segments: List[Dict[str, Any]],
    idx: int,
    keep_spk: List[str],
) -> str | None:
    n = len(segments)
    target_idx = idx

    prev_seg = segments[target_idx - 1] if target_idx > 0 else None
    next_seg = segments[target_idx + 1] if target_idx + 1 < n else None

    candidates: List[Tuple[float, str]] = []

    for neighbor in (prev_seg, next_seg):
        if not neighbor:
            continue
        spk = neighbor.get("speaker")
        if spk not in keep_spk:
            continue

        try:
            s0 = float(segments[target_idx].get("start", 0.0) or 0.0)
            e0 = float(segments[target_idx].get("end", s0) or s0)
            s1 = float(neighbor.get("start", 0.0) or 0.0)
            e1 = float(neighbor.get("end", s1) or s1)
        except (TypeError, ValueError):
            continue

        gap = max(0.0, max(s0 - e1, s1 - e0))
        score = 1.0 / (1.0 + gap)
        candidates.append((score, spk))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[0])[1]


def reassign_dropped_speakers(
    segments: List[Dict[str, Any]],
    keep_spk: List[str],
    drop_spk: List[str],
) -> List[Dict[str, Any]]:
    if not segments or not drop_spk:
        return segments

    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0) or 0.0))
    for idx, seg in enumerate(segs):
        spk = seg.get("speaker")
        if spk not in drop_spk:
            continue

        target = _nearest_keep_speaker(segs, idx, keep_spk)
        if target is None:
            # 周辺に keep スピーカーがいない場合は最長話者に吸収させる
            target = keep_spk[0]

        seg["speaker"] = target

    return segs


def merge_timeline(
    segments: List[Dict[str, Any]],
    max_gap: float = 0.3,
    min_len: float = 0.0,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    segs = sorted(segments, key=lambda s: float(s.get("start", 0.0) or 0.0))
    merged: List[Dict[str, Any]] = []

    prev = segs[0].copy()
    for cur in segs[1:]:
        try:
            prev_start = float(prev.get("start", 0.0) or 0.0)
            prev_end = float(prev.get("end", prev_start) or prev_start)
            cur_start = float(cur.get("start", 0.0) or 0.0)
            cur_end = float(cur.get("end", cur_start) or cur_start)
        except (TypeError, ValueError):
            merged.append(prev)
            prev = cur.copy()
            continue

        if cur.get("speaker") == prev.get("speaker"):
            gap = cur_start - prev_end
            if 0.0 <= gap <= max_gap:
                prev["end"] = max(prev_end, cur_end)
                continue

        dur = prev_end - prev_start
        if dur >= min_len:
            merged.append(prev)
        else:
            merged.append(prev)

        prev = cur.copy()

    merged.append(prev)
    return merged


def sanitize_speaker_timeline(
    segments: List[Dict[str, Any]],
    duration_sec: float,
    config: Any,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    stats = collect_speaker_stats(segments)
    coverage_th = getattr(config, "coverage_threshold", 0.97)
    merge_gap = getattr(config, "merge_gap_sec", 0.3)
    min_len = getattr(config, "min_segment_len_sec", 0.0)
    max_speakers = getattr(config, "max_speakers", None)

    keep_spk, drop_spk = infer_effective_speakers(
        stats,
        coverage_th=coverage_th,
        max_speakers=max_speakers,
    )

    LOGGER.info(
        "[POST-DIAR] stats=%s, keep=%s, drop=%s",
        {k: round(v["duration"], 2) for k, v in stats.items()},
        keep_spk,
        drop_spk,
    )

    segs = segments
    if drop_spk:
        segs = reassign_dropped_speakers(segs, keep_spk, drop_spk)

    segs = merge_timeline(segs, max_gap=merge_gap, min_len=min_len)

    # 時間範囲のサニティチェック
    fixed: List[Dict[str, Any]] = []
    for seg in segs:
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except (TypeError, ValueError):
            continue

        start = max(0.0, min(start, duration_sec))
        end = max(start, min(end, duration_sec))

        out = dict(seg)
        out["start"] = start
        out["end"] = end
        fixed.append(out)

    return fixed
