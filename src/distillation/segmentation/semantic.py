"""
Embedding-based semantic segmenter.

Algorithm:
  1. Encode each TranscriptSegment text with a sentence-transformer.
  2. Compute cosine similarity between consecutive segment embeddings.
  3. A low similarity (below threshold) signals a topic boundary.
  4. Group TranscriptSegments between boundaries into Segments.
  5. Merge short Segments to honour min_clip_duration.

Why not use an LLM for segmentation?
  - Embedding-based segmentation is O(n) and parallelisable; LLM calls are
    slow and expensive for long lectures (1-hour = ~3,000 segments).
  - The LLM's role is scoring *candidate* segments, not finding boundaries —
    it adds the most value on semantic quality evaluation, not detection.
  - TextTiling / cosine-distance segmentation is well understood; we can
    tune the threshold on a labelled dataset without touching the LLM layer.

Calibration guidance (see docs/architecture.md):
  - threshold ~0.35 works well for English lectures.
  - For Arabic/English code-switching (khutbas), try 0.45 — embeddings from
    multilingual-e5 are more reliable than MiniLM for mixed-language text.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from distillation.config import Config
from distillation.models import Segment, Transcript, TranscriptSegment
from distillation.segmentation.base import BaseSegmenter

logger = logging.getLogger(__name__)


class SemanticSegmenter(BaseSegmenter):
    """
    Segments a transcript by detecting embedding-space topic shifts.

    For khutbas, consider using:
        embedding_model = "intfloat/multilingual-e5-small"
    which handles Arabic and code-switching better than MiniLM.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._encoder = None  # lazy load

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence encoder: %s", self.cfg.embedding_model)
            self._encoder = SentenceTransformer(self.cfg.embedding_model)
        return self._encoder

    def segment(self, transcript: Transcript) -> list[Segment]:
        ts_segs = transcript.segments
        if not ts_segs:
            return []

        # ── 1. Embed all transcript segments ──────────────────────────────────
        encoder = self._get_encoder()
        texts = [s.text for s in ts_segs]
        embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False)
        # embeddings: (n, dim)

        # ── 2. Compute consecutive cosine similarities ─────────────────────────
        similarities = self._consecutive_similarities(embeddings)

        # ── 3. Find boundaries where similarity drops below threshold ──────────
        threshold = self.cfg.semantic_similarity_threshold
        # boundaries[i] = True means a new segment starts at ts_segs[i+1]
        boundaries: list[int] = [0]   # always start a segment at index 0
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1)
        boundaries.append(len(ts_segs))   # sentinel end

        logger.info(
            "Found %d topic boundaries (threshold=%.2f)",
            len(boundaries) - 2, threshold
        )

        # ── 4. Build Segments from boundary ranges ─────────────────────────────
        raw_segments: list[Segment] = []
        seg_id = 0
        for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
            chunk = ts_segs[start_idx:end_idx]
            if not chunk:
                continue
            raw_segments.append(self._build_segment(seg_id, chunk))
            seg_id += 1

        # ── 5. Merge short segments ────────────────────────────────────────────
        merged = self._merge_short_segments(raw_segments, ts_segs)
        logger.info("SemanticSegmenter produced %d segments", len(merged))
        return merged

    @staticmethod
    def _consecutive_similarities(embeddings: np.ndarray) -> list[float]:
        """Cosine similarity between each consecutive pair of embeddings."""
        if len(embeddings) < 2:
            return []
        # Vectorised pairwise similarity; we only need the diagonal offset by 1
        sims = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i].reshape(1, -1)
            b = embeddings[i + 1].reshape(1, -1)
            sims.append(float(cosine_similarity(a, b)[0, 0]))
        return sims

    def _merge_short_segments(
        self,
        segments: list[Segment],
        ts_segs: list[TranscriptSegment],
    ) -> list[Segment]:
        """
        Merge adjacent segments that are shorter than min_clip_duration.

        We merge forward (short segment absorbs the next one) so boundary
        detection results are preserved as much as possible.
        """
        min_dur = self.cfg.min_clip_duration
        result: list[Segment] = []
        buffer_ids: list[int] = []

        for seg in segments:
            buffer_ids.extend(seg.transcript_segment_ids)
            buffer_ts = [s for s in ts_segs if s.id in set(buffer_ids)]
            current_dur = buffer_ts[-1].end - buffer_ts[0].start if buffer_ts else 0

            if current_dur >= min_dur:
                result.append(self._build_segment(len(result), buffer_ts))
                buffer_ids = []

        # Flush remaining
        if buffer_ids:
            buffer_ts = [s for s in ts_segs if s.id in set(buffer_ids)]
            if buffer_ts:
                if result:
                    # Absorb into the last segment rather than creating a
                    # tiny trailing segment
                    merged_ids = result[-1].transcript_segment_ids + buffer_ids
                    merged_ts = [s for s in ts_segs if s.id in set(merged_ids)]
                    result[-1] = self._build_segment(result[-1].segment_id, merged_ts)
                else:
                    result.append(self._build_segment(0, buffer_ts))

        return result
