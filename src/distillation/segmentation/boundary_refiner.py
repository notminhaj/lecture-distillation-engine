"""
Post-processing boundary refinement for segmentation output.

Runs four ordered passes on raw segments to improve clip boundaries:
  1. Snap boundaries to sentence (TranscriptSegment) edges.
  2. Snap boundaries to the largest inter-word silence gap.
  3. Protect strong hook openings at segment starts.
  4. Enforce min/max duration constraints via merge/split.

This module is segmenter-agnostic — it works identically after
SemanticSegmenter or SlidingWindowSegmenter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from distillation.models import Segment, Transcript, TranscriptSegment, Word

if TYPE_CHECKING:
    from distillation.config import Config

logger = logging.getLogger(__name__)


class BoundaryRefiner:
    """Refine segment boundaries using transcript structure and word-level timing."""

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def refine(self, segments: list[Segment], transcript: Transcript) -> list[Segment]:
        """Apply all refinement passes in order."""
        if not self.cfg.enable_boundary_refinement or not segments:
            return segments

        ts_segs = transcript.segments
        if not ts_segs:
            return segments

        refined = self._snap_to_sentence_boundaries(segments, ts_segs)
        refined = self._snap_to_pauses(refined, ts_segs)
        refined = self._protect_hooks(refined, ts_segs)
        refined = self._enforce_duration_bounds(refined, ts_segs)
        return refined

    # ── Pass 1: Sentence-aware boundary snapping ────────────────────────────

    def _snap_to_sentence_boundaries(
        self,
        segments: list[Segment],
        ts_segs: list[TranscriptSegment],
    ) -> list[Segment]:
        """Snap each boundary to the nearest TranscriptSegment edge within ±window."""
        window = self.cfg.sentence_boundary_window
        ts_starts = [ts.start for ts in ts_segs]
        ts_ends = [ts.end for ts in ts_segs]

        new_boundaries: list[tuple[float, float]] = []
        for seg in segments:
            snapped_start = self._nearest_value(ts_starts, seg.start, window)
            snapped_end = self._nearest_value(ts_ends, seg.end, window)
            # Ensure start < end
            if snapped_start >= snapped_end:
                snapped_start = seg.start
                snapped_end = seg.end
            new_boundaries.append((snapped_start, snapped_end))

        # Fix overlaps: ensure seg[i].end <= seg[i+1].start
        for i in range(len(new_boundaries) - 1):
            s_cur, e_cur = new_boundaries[i]
            s_next, e_next = new_boundaries[i + 1]
            if e_cur > s_next:
                # Move boundary to midpoint
                mid = (e_cur + s_next) / 2
                new_boundaries[i] = (s_cur, mid)
                new_boundaries[i + 1] = (mid, e_next)

        return self._rebuild_segments(new_boundaries, ts_segs)

    # ── Pass 2: Pause-based boundary refinement ─────────────────────────────

    def _snap_to_pauses(
        self,
        segments: list[Segment],
        ts_segs: list[TranscriptSegment],
    ) -> list[Segment]:
        """Shift boundaries between consecutive segments to the largest inter-word gap."""
        if len(segments) < 2:
            return segments

        all_words = self._collect_words(ts_segs)
        if not all_words:
            return segments

        threshold_s = self.cfg.silence_threshold_ms / 1000.0
        window = self.cfg.sentence_boundary_window

        new_boundaries: list[tuple[float, float]] = [
            (seg.start, seg.end) for seg in segments
        ]

        for i in range(len(segments) - 1):
            boundary_time = segments[i].end
            # Find words near the boundary
            nearby_words = [
                w for w in all_words
                if boundary_time - window <= w.start <= boundary_time + window
            ]
            if len(nearby_words) < 2:
                continue

            nearby_words.sort(key=lambda w: w.start)

            # Find the largest gap
            best_gap = 0.0
            best_gap_time = boundary_time
            for j in range(len(nearby_words) - 1):
                gap = nearby_words[j + 1].start - nearby_words[j].end
                if gap > best_gap:
                    best_gap = gap
                    best_gap_time = (nearby_words[j].end + nearby_words[j + 1].start) / 2

            if best_gap >= threshold_s:
                s_cur, _ = new_boundaries[i]
                _, e_next = new_boundaries[i + 1]
                new_boundaries[i] = (s_cur, best_gap_time)
                new_boundaries[i + 1] = (best_gap_time, e_next)

        return self._rebuild_segments(new_boundaries, ts_segs)

    # ── Pass 3: Hook protection ─────────────────────────────────────────────

    def _protect_hooks(
        self,
        segments: list[Segment],
        ts_segs: list[TranscriptSegment],
    ) -> list[Segment]:
        """Keep strong hook openings with their segment; shift bleed backward."""
        if len(segments) < 2:
            return segments

        all_words = self._collect_words(ts_segs)
        if not all_words:
            return segments

        hook_window = self.cfg.hook_protection_window

        new_boundaries: list[tuple[float, float]] = [
            (seg.start, seg.end) for seg in segments
        ]

        for i in range(1, len(segments)):
            seg_start = segments[i].start
            prev_end = segments[i - 1].end

            # Words in first hook_window seconds of current segment
            first_words = [
                w for w in all_words
                if seg_start <= w.start < seg_start + hook_window
            ]
            # Words in last hook_window seconds of previous segment
            last_words = [
                w for w in all_words
                if prev_end - hook_window <= w.start < prev_end
            ]

            first_wps = len(first_words) / hook_window if hook_window > 0 else 0
            last_wps = len(last_words) / hook_window if hook_window > 0 else 0

            # Check for pause before the hook
            gap_before_hook = 0.0
            if first_words:
                first_words.sort(key=lambda w: w.start)
                # Find last word before the boundary
                words_before = [w for w in all_words if w.end <= seg_start + 0.5 and w.end >= seg_start - 1.0]
                if words_before:
                    words_before.sort(key=lambda w: w.end)
                    gap_before_hook = first_words[0].start - words_before[-1].end

            # If first-3s WPS is similar to prev end AND no pause → shift backward
            if (last_wps > 0 and first_wps > 0
                    and first_wps < last_wps * 1.5
                    and gap_before_hook < 0.3):
                # Shift boundary backward by one TranscriptSegment
                prev_ts_ids = segments[i - 1].transcript_segment_ids
                if len(prev_ts_ids) > 1:
                    # Find the TS that contains the boundary area
                    last_ts_id = prev_ts_ids[-1]
                    last_ts = next((ts for ts in ts_segs if ts.id == last_ts_id), None)
                    if last_ts:
                        new_boundary = last_ts.start
                        s_prev, _ = new_boundaries[i - 1]
                        _, e_cur = new_boundaries[i]
                        if new_boundary > s_prev:
                            new_boundaries[i - 1] = (s_prev, new_boundary)
                            new_boundaries[i] = (new_boundary, e_cur)

        return self._rebuild_segments(new_boundaries, ts_segs)

    # ── Pass 4: Duration enforcement ────────────────────────────────────────

    def _enforce_duration_bounds(
        self,
        segments: list[Segment],
        ts_segs: list[TranscriptSegment],
    ) -> list[Segment]:
        """Merge short segments, split long ones."""
        min_dur = self.cfg.min_clip_duration
        max_dur = self.cfg.max_clip_duration
        all_words = self._collect_words(ts_segs)

        # Merge short segments
        merged: list[Segment] = []
        for seg in segments:
            if merged and merged[-1].duration + seg.duration < max_dur:
                # If previous segment is too short, merge
                if merged[-1].duration < min_dur:
                    merged[-1] = self._merge_two_segments(
                        len(merged) - 1, merged[-1], seg, ts_segs
                    )
                    continue
            # If current segment is too short and we have a previous segment
            if seg.duration < min_dur and merged:
                prev = merged[-1]
                # Merge with the shorter neighbor (only have previous at this point)
                if prev.duration + seg.duration <= max_dur:
                    merged[-1] = self._merge_two_segments(
                        len(merged) - 1, prev, seg, ts_segs
                    )
                    continue
            merged.append(seg)

        # Handle last segment being too short
        if len(merged) > 1 and merged[-1].duration < min_dur:
            last = merged.pop()
            if merged[-1].duration + last.duration <= max_dur:
                merged[-1] = self._merge_two_segments(
                    len(merged) - 1, merged[-1], last, ts_segs
                )
            else:
                merged.append(last)

        # Split long segments
        result: list[Segment] = []
        for seg in merged:
            if seg.duration > max_dur and all_words:
                split = self._split_at_pause(seg, ts_segs, all_words, min_dur)
                result.extend(split)
            else:
                result.append(seg)

        # Re-index
        for i, seg in enumerate(result):
            seg.segment_id = i

        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_value(values: list[float], target: float, window: float) -> float:
        """Return the value in *values* nearest to *target* within ±window."""
        best = target
        best_dist = window + 1
        for v in values:
            dist = abs(v - target)
            if dist <= window and dist < best_dist:
                best = v
                best_dist = dist
        return best

    @staticmethod
    def _collect_words(ts_segs: list[TranscriptSegment]) -> list[Word]:
        """Flatten all words from all TranscriptSegments."""
        words: list[Word] = []
        for ts in ts_segs:
            words.extend(ts.words)
        return words

    def _rebuild_segments(
        self,
        boundaries: list[tuple[float, float]],
        ts_segs: list[TranscriptSegment],
    ) -> list[Segment]:
        """Rebuild Segment objects from (start, end) boundaries and TranscriptSegments."""
        result: list[Segment] = []
        for i, (start, end) in enumerate(boundaries):
            # Find TranscriptSegments whose midpoint falls within this boundary
            contained = [
                ts for ts in ts_segs
                if (ts.start + ts.end) / 2 >= start and (ts.start + ts.end) / 2 < end
            ]
            if not contained:
                # Fallback: include any TS that overlaps
                contained = [
                    ts for ts in ts_segs
                    if ts.start < end and ts.end > start
                ]
            if contained:
                contained.sort(key=lambda ts: ts.start)
                result.append(Segment(
                    segment_id=i,
                    start=contained[0].start,
                    end=contained[-1].end,
                    text=" ".join(ts.text.strip() for ts in contained),
                    transcript_segment_ids=[ts.id for ts in contained],
                ))
        return result

    @staticmethod
    def _merge_two_segments(
        new_id: int,
        a: Segment,
        b: Segment,
        ts_segs: list[TranscriptSegment],
    ) -> Segment:
        """Merge two adjacent segments into one."""
        merged_ids = a.transcript_segment_ids + b.transcript_segment_ids
        merged_ts = [ts for ts in ts_segs if ts.id in set(merged_ids)]
        merged_ts.sort(key=lambda ts: ts.start)
        return Segment(
            segment_id=new_id,
            start=merged_ts[0].start,
            end=merged_ts[-1].end,
            text=" ".join(ts.text.strip() for ts in merged_ts),
            transcript_segment_ids=[ts.id for ts in merged_ts],
        )

    def _split_at_pause(
        self,
        seg: Segment,
        ts_segs: list[TranscriptSegment],
        all_words: list[Word],
        min_dur: float,
    ) -> list[Segment]:
        """Split a segment at the largest internal pause, if both halves meet min_dur."""
        seg_words = [
            w for w in all_words
            if w.start >= seg.start and w.end <= seg.end
        ]
        if len(seg_words) < 2:
            return [seg]

        seg_words.sort(key=lambda w: w.start)

        # Find the best split point (largest gap, both halves >= min_dur)
        best_gap = 0.0
        best_split_time = 0.0
        for j in range(len(seg_words) - 1):
            gap = seg_words[j + 1].start - seg_words[j].end
            split_time = (seg_words[j].end + seg_words[j + 1].start) / 2
            left_dur = split_time - seg.start
            right_dur = seg.end - split_time
            if left_dur >= min_dur and right_dur >= min_dur and gap > best_gap:
                best_gap = gap
                best_split_time = split_time

        if best_split_time == 0.0:
            return [seg]

        # Build two segments from the split
        seg_ts = [ts for ts in ts_segs if ts.id in set(seg.transcript_segment_ids)]
        seg_ts.sort(key=lambda ts: ts.start)

        left_ts = [ts for ts in seg_ts if (ts.start + ts.end) / 2 < best_split_time]
        right_ts = [ts for ts in seg_ts if (ts.start + ts.end) / 2 >= best_split_time]

        result: list[Segment] = []
        if left_ts:
            result.append(Segment(
                segment_id=0,
                start=left_ts[0].start,
                end=left_ts[-1].end,
                text=" ".join(ts.text.strip() for ts in left_ts),
                transcript_segment_ids=[ts.id for ts in left_ts],
            ))
        if right_ts:
            result.append(Segment(
                segment_id=0,
                start=right_ts[0].start,
                end=right_ts[-1].end,
                text=" ".join(ts.text.strip() for ts in right_ts),
                transcript_segment_ids=[ts.id for ts in right_ts],
            ))

        if not result:
            return [seg]
        return result
