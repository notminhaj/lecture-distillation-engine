"""Tests for the clip selector."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from distillation.models import Domain, EngagementAxes, Segment, ScoredSegment
from distillation.selection.selector import ClipSelector, DOMAIN_WEIGHTS


def make_scored_segment_with_text(
    segment_id: int,
    start: float,
    end: float,
    text: str,
    score_override: float = 0.7,
    *,
    narrative_completeness: float | None = None,
    standalone_coherence: float | None = None,
) -> ScoredSegment:
    """Variant of make_scored_segment that sets custom text on the segment."""
    nc = narrative_completeness if narrative_completeness is not None else score_override
    sc = standalone_coherence if standalone_coherence is not None else score_override
    axes = EngagementAxes(
        semantic_density=score_override,
        emotional_resonance=score_override,
        standalone_coherence=sc,
        narrative_completeness=nc,
        domain_integrity=score_override,
        hook_strength=score_override,
    )
    seg = Segment(
        segment_id=segment_id,
        start=start,
        end=end,
        text=text,
        transcript_segment_ids=[segment_id],
    )
    return ScoredSegment(segment=seg, scores=axes)


def make_scored_segment(
    segment_id: int,
    start: float,
    end: float,
    score_override: float = 0.5,
    *,
    narrative_completeness: float | None = None,
    standalone_coherence: float | None = None,
) -> ScoredSegment:
    nc = narrative_completeness if narrative_completeness is not None else score_override
    sc = standalone_coherence if standalone_coherence is not None else score_override
    axes = EngagementAxes(
        semantic_density=score_override,
        emotional_resonance=score_override,
        standalone_coherence=sc,
        narrative_completeness=nc,
        domain_integrity=score_override,
        hook_strength=score_override,
    )
    seg = Segment(
        segment_id=segment_id,
        start=start,
        end=end,
        text=f"Segment {segment_id} text.",
        transcript_segment_ids=[segment_id],
    )
    return ScoredSegment(segment=seg, scores=axes)


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.min_clip_duration = 30.0
    cfg.max_clip_duration = 90.0
    cfg.target_clip_count = 3
    cfg.min_narrative_completeness = 0.5
    return cfg


@pytest.fixture
def selector(config):
    return ClipSelector(config)


class TestClipSelector:
    def test_returns_at_most_target_count(self, selector):
        segments = [
            make_scored_segment(i, i * 60.0, i * 60.0 + 45.0)
            for i in range(10)
        ]
        clips = selector.select(segments, domain=Domain.KHUTBA)
        assert len(clips) <= 3

    def test_clips_are_non_overlapping(self, selector):
        segments = [
            make_scored_segment(i, i * 60.0, i * 60.0 + 50.0)
            for i in range(10)
        ]
        clips = selector.select(segments, domain=Domain.GENERIC)
        # Check no two clips overlap
        for i, a in enumerate(clips):
            for b in clips[i + 1:]:
                assert not (a.start < b.end and a.end > b.start), (
                    f"Clips {a.clip_id} and {b.clip_id} overlap"
                )

    def test_clips_ordered_by_timeline(self, selector):
        segments = [
            make_scored_segment(i, i * 60.0, i * 60.0 + 45.0)
            for i in range(6)
        ]
        clips = selector.select(segments)
        starts = [c.start for c in clips]
        assert starts == sorted(starts)

    def test_filters_by_min_duration(self, selector):
        """Segments shorter than min_clip_duration should be excluded."""
        short = make_scored_segment(0, 0.0, 10.0, score_override=0.99)  # too short
        long = make_scored_segment(1, 60.0, 110.0, score_override=0.50)
        # long is within [30, 90]? 50s → yes
        clips = selector.select([short, long])
        clip_segment_ids = [c.scored_segment.segment.segment_id for c in clips]
        assert 0 not in clip_segment_ids  # short excluded

    def test_empty_input_returns_empty(self, selector):
        assert selector.select([]) == []

    def test_highest_scoring_segment_selected(self, selector):
        """When segments don't overlap, the top scorer should always be chosen."""
        segments = [
            make_scored_segment(0, 0.0, 45.0, score_override=0.9),
            make_scored_segment(1, 60.0, 105.0, score_override=0.3),
            make_scored_segment(2, 120.0, 165.0, score_override=0.1),
        ]
        clips = selector.select(segments, domain=Domain.GENERIC)
        # First clip by score (seg 0 = 0.9) should appear
        selected_ids = {c.scored_segment.segment.segment_id for c in clips}
        assert 0 in selected_ids

    def test_completeness_gate_filters_at_threshold(self, config):
        """nc=0.45 is filtered out; nc=0.55 passes through."""
        selector = ClipSelector(config)
        below = make_scored_segment(0, 0.0, 45.0, score_override=0.9, narrative_completeness=0.45)
        above = make_scored_segment(1, 60.0, 105.0, score_override=0.5, narrative_completeness=0.55)
        clips = selector.select([below, above], domain=Domain.GENERIC)
        selected_ids = {c.scored_segment.segment.segment_id for c in clips}
        assert 0 not in selected_ids, "nc=0.45 should be filtered by completeness gate"
        assert 1 in selected_ids, "nc=0.55 should pass the completeness gate"

    def test_merge_trigger_fires_on_low_standalone_coherence(self, config):
        """A segment with nc=0.6 but sc=0.3 should generate a merge candidate."""
        selector = ClipSelector(config)
        # Segment A: nc fine, sc weak — should trigger merge
        seg_a = make_scored_segment(
            0, 0.0, 40.0, score_override=0.6,
            narrative_completeness=0.6,
            standalone_coherence=0.3,
        )
        # Segment B: adjacent, fine scores
        seg_b = make_scored_segment(
            1, 40.0, 75.0, score_override=0.6,
            narrative_completeness=0.6,
            standalone_coherence=0.7,
        )
        merged = selector._generate_merge_candidates([seg_a, seg_b])
        assert len(merged) == 1, (
            "Expected exactly one merge candidate when sc < 0.5 on first segment"
        )
        # Verify the sc bonus was applied
        assert merged[0].scores.standalone_coherence > (0.3 + 0.7) / 2, (
            "Merged sc should include the 0.10 bonus for the weak sc trigger"
        )

    def test_khutba_weights_sum_to_one(self):
        """Regression guard: DOMAIN_WEIGHTS[Domain.KHUTBA] must sum to exactly 1.0."""
        weights = DOMAIN_WEIGHTS[Domain.KHUTBA]
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, (
            f"KHUTBA weights sum to {total:.4f}, expected 1.0"
        )


class TestReferentialOpenerPenalty:
    """
    Regression tests for the ns-002 additions to _referential_opener_penalty().

    Each test verifies that a specific pattern returns (0.5, 0.5) — capping
    standalone_coherence and narrative_completeness at 0.5 — rather than the
    no-cap default of (1.0, 1.0).
    """

    def test_past_perfect_we_had_mentioned_triggers_penalty(self):
        """'we had mentioned' is a past-perfect referential opener → sc/nc capped at 0.5."""
        text = "we had mentioned that the reason why Abyssinia was chosen is because of faith."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5, got {sc_cap} for past-perfect opener"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5, got {nc_cap} for past-perfect opener"

    def test_past_perfect_he_had_said_triggers_penalty(self):
        """'he had said' pattern should also fire the past-perfect referential rule."""
        text = "he had said that this was the teaching of the Prophet, so we continue from there."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5, got {sc_cap} for 'he had said'"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5, got {nc_cap} for 'he had said'"

    def test_continuation_conjunction_and_triggers_penalty(self):
        """Segment opening with 'and' (lowercase) → capped at 0.5 (continuation-conjunction)."""
        text = "and so this is what ibn Taymiya argued about the verse of surah al-Hajj."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for conjunction opener 'and', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for conjunction opener 'and', got {nc_cap}"

    def test_continuation_conjunction_however_triggers_penalty(self):
        """'however' at the start of a segment is a continuation-conjunction opener."""
        text = "however the scholars disagreed on the interpretation of this particular verse."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'however' opener, got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'however' opener, got {nc_cap}"

    def test_mid_comparison_like_all_triggers_penalty(self):
        """'like all the prophets' is a mid-comparison opener → sc/nc capped at 0.5."""
        text = "like all the prophets but he cannot persist in them and will repent immediately."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'like all' opener, got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'like all' opener, got {nc_cap}"

    def test_clean_opener_no_penalty(self):
        """A segment with a clean opener should return (1.0, 1.0) — no cap."""
        text = "The Prophet peace be upon him said: seek knowledge even unto China."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 1.0, f"Expected sc_cap=1.0 for clean opener, got {sc_cap}"
        assert nc_cap == 1.0, f"Expected nc_cap=1.0 for clean opener, got {nc_cap}"

    # ── ns-011 additions: new referential opener patterns ────────────────────

    def test_as_i_stated_triggers_penalty(self):
        """'as I stated' (new: 'stated' added to subject pattern) → sc/nc capped at 0.5."""
        text = "as I stated earlier, the Prophet had a unique approach to this matter."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'as I stated', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'as I stated', got {nc_cap}"

    def test_as_stated_passive_triggers_penalty(self):
        """'as stated' (passive, no subject) → sc/nc capped at 0.5."""
        text = "as stated in the previous section, this ruling applies to all Muslims."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'as stated', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'as stated', got {nc_cap}"

    def test_as_described_passive_triggers_penalty(self):
        """'as described' (passive, no subject) → sc/nc capped at 0.5."""
        text = "as described above, the scholars had three different positions on this."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'as described', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'as described', got {nc_cap}"

    def test_subject_less_i_mentioned_opener_triggers_penalty(self):
        """'I mentioned' at the start (without 'as') → sc/nc capped at 0.5."""
        text = "I mentioned this hadith last week, so let me now explain the ruling that follows."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'I mentioned' opener, got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'I mentioned' opener, got {nc_cap}"

    def test_subject_less_we_said_opener_triggers_penalty(self):
        """'we said' at the start (without 'as') → sc/nc capped at 0.5."""
        text = "we said that the first school of thought holds that this verse is general."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'we said' opener, got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'we said' opener, got {nc_cap}"

    def test_building_on_that_triggers_penalty(self):
        """'building on that' is a continuation marker → sc/nc capped at 0.5."""
        text = "building on that point, we can now understand why ibn Taymiya disagreed."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'building on that', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'building on that', got {nc_cap}"


class TestStructuralCompletenessPenalty:
    """
    Regression tests for the ns-002 additions to _structural_completeness_penalty().

    Covers: _CONTINUATION_STARTERS frozenset producing -0.25 penalty (stronger than
    the generic -0.15 for other lowercase starters).
    """

    def _make_ss(self, text: str) -> ScoredSegment:
        """Minimal ScoredSegment wrapper with just the text needed for the penalty."""
        axes = EngagementAxes(
            semantic_density=0.5, emotional_resonance=0.5, standalone_coherence=0.5,
            narrative_completeness=0.5, domain_integrity=0.5, hook_strength=0.5,
        )
        seg = Segment(segment_id=0, start=0.0, end=45.0, text=text, transcript_segment_ids=[0])
        return ScoredSegment(segment=seg, scores=axes)

    def test_continuation_starter_and_receives_strong_penalty(self):
        """'and' at start → _CONTINUATION_STARTERS match → penalty = -0.25 (not -0.15).

        Combined with missing terminal punctuation (-0.15), total would be -0.40,
        but the clamp at 0.7 applies → final multiplier is 0.7.
        """
        # No terminal punctuation → two penalties fire: -0.25 (continuation) + -0.15 (no punct)
        ss = self._make_ss("and so as you know this was the lesson from the hadith")
        mult = ClipSelector._structural_completeness_penalty(ss)
        # 1.0 - 0.25 - 0.15 = 0.60, but clamped to max(0.7, 0.60) = 0.7
        assert mult == 0.7, f"Expected multiplier 0.7 (clamped), got {mult}"

    def test_continuation_starter_immensely_receives_strong_penalty(self):
        """'immensely' (in _CONTINUATION_STARTERS) → -0.25 penalty, not -0.15."""
        # Ends with period → only one penalty fires (-0.25 for continuation starter)
        ss = self._make_ss("immensely and that is sheikh al-Islam ibn Taymiya.")
        mult = ClipSelector._structural_completeness_penalty(ss)
        # 1.0 - 0.25 (continuation) = 0.75; no terminal punct penalty since ends with '.'
        assert mult == 0.75, f"Expected multiplier 0.75 for continuation starter, got {mult}"

    def test_regular_lowercase_start_receives_lighter_penalty(self):
        """A lowercase non-continuation-starter word → -0.15 penalty (not -0.25)."""
        # 'the' is not in _CONTINUATION_STARTERS → generic -0.15 penalty
        ss = self._make_ss("the Prophet said seek knowledge even unto China.")
        mult = ClipSelector._structural_completeness_penalty(ss)
        # 1.0 - 0.15 (lowercase non-continuation start) = 0.85; ends with '.' → no punct penalty
        assert mult == 0.85, f"Expected multiplier 0.85 for lowercase non-continuation start, got {mult}"

    def test_uppercase_clean_segment_no_penalty(self):
        """Uppercase start + terminal punctuation → multiplier 1.0 (no penalty)."""
        ss = self._make_ss("The Prophet peace be upon him said: seek knowledge even unto China.")
        mult = ClipSelector._structural_completeness_penalty(ss)
        assert mult == 1.0, f"Expected multiplier 1.0 for clean segment, got {mult}"

    # ── ns-012 additions: expanded continuation starters ───────────────────

    def test_continuation_starter_that_receives_strong_penalty(self):
        """'that' is now a continuation starter → -0.25 penalty."""
        ss = self._make_ss("that Jibril is coming this is exactly the claims.")
        mult = ClipSelector._structural_completeness_penalty(ss)
        assert mult == 0.75, f"Expected 0.75 for 'that' continuation, got {mult}"

    def test_continuation_starter_when_receives_strong_penalty(self):
        """'when' is now a continuation starter → -0.25 penalty."""
        ss = self._make_ss("when Amr wants to kill the envoy and this love.")
        mult = ClipSelector._structural_completeness_penalty(ss)
        assert mult == 0.75, f"Expected 0.75 for 'when' continuation, got {mult}"

    def test_continuation_starter_as_receives_strong_penalty(self):
        """'as' is now a continuation starter → -0.25 penalty."""
        ss = self._make_ss("as well that that is O Messenger of Allah.")
        mult = ClipSelector._structural_completeness_penalty(ss)
        assert mult == 0.75, f"Expected 0.75 for 'as' continuation, got {mult}"


class TestReferentialOpenerPatterns:
    """Tests for ns-012 referential opener pattern additions."""

    def test_as_well_triggers_referential_penalty(self):
        """'as well that that...' → referential opener detected → capped at 0.5."""
        text = "as well that that is O Messenger of Allah can my sins be forgiven"
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'as well', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'as well', got {nc_cap}"

    def test_that_is_why_triggers_referential_penalty(self):
        """'that is why...' → referential opener detected → capped at 0.5."""
        text = "that is why we must reflect on this lesson carefully"
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'that is why', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'that is why', got {nc_cap}"

    def test_when_he_triggers_referential_penalty(self):
        """'when he arrived...' → referential opener detected → capped at 0.5."""
        text = "when he arrived in Madinah the Prophet was already there"
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'when he', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'when he', got {nc_cap}"

    def test_which_means_triggers_referential_penalty(self):
        """'which means...' → referential opener detected → capped at 0.5."""
        text = "which means the ruling only applies in specific circumstances"
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 0.5, f"Expected sc_cap=0.5 for 'which means', got {sc_cap}"
        assert nc_cap == 0.5, f"Expected nc_cap=0.5 for 'which means', got {nc_cap}"

    def test_clean_when_question_not_penalised(self):
        """'When did the Prophet...' (uppercase, question form) should not be penalised."""
        text = "When did the Prophet first receive revelation? This is a critical question."
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        assert sc_cap == 1.0, f"Expected sc_cap=1.0 for uppercase 'When' question, got {sc_cap}"
