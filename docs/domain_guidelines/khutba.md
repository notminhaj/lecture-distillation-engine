# Domain Guidelines: Islamic Khutba (Sermon)

This document guides prompt engineering and scoring calibration for khutba content.
It should be reviewed by a domain expert (Islamic scholar or senior da'wah practitioner).

## What Makes a Good Khutba Clip

### High-quality clip characteristics
1. **Complete argument**: The speaker makes a point, provides evidence (Quran/hadith),
   and draws a lesson. A clip that cuts mid-proof is incomplete.
2. **Attribution present**: If a hadith is cited, attribution should be audible in
   the clip ("The Prophet ﷺ said...") even if partial.
3. **Standalone reminder**: A Muslim viewer who missed the full khutba should gain
   a complete reminder from the clip.
4. **Emotional hook**: Clips that open with a question ("Have you ever asked yourself...")
   or a striking Quranic image tend to retain viewers.

### Clips to avoid
- Mid-ruling cuts (e.g., "...and therefore it is [clip ends]")
- Arabic-heavy passages without English translation (for English-medium khutbas)
- Clips that require the previous 5 minutes of argument to make sense
- Personal anecdotes without the lesson/application that follows

## Common Arabic Terms (non-foreign to audience)
The LLM should not penalise these for "jargon" — they are part of the domain vocabulary:
- Subhanallah (Glory be to God)
- Alhamdulillah (All praise is to God)
- Astaghfirullah (I seek God's forgiveness)
- Salallahu alayhi wa salam / ﷺ (peace be upon him — Prophet's honorific)
- Bismillah (In the name of God)
- Inshallah (If God wills)
- Hadith (saying/tradition of the Prophet)
- Sunnah (practice/example of the Prophet)
- Fiqh (Islamic jurisprudence)
- Ummah (Muslim community)
- Deen (the religion; way of life)

## Scoring Calibration Notes

### `domain_integrity` for khutba
Score LOW if:
- A hadith is cited without attribution
- A ruling is stated without its basis (Quran/hadith reference)
- The clip implies something is haram/halal without completing the argument

Score HIGH if:
- Quran/hadith citations are complete (surah name, hadith source, or explicit "Prophet said")
- The ruling and its basis are both in the clip
- No theological ambiguity is left unresolved

### `standalone_coherence` for khutba
Score LOW if:
- The speaker says "as I mentioned earlier" or "going back to..."
- The argument depends on a definition given in the first half of the khutba
- Without context, a viewer might draw a wrong theological conclusion

### `hook_strength` for khutba
Score HIGH if the first 3 seconds contain:
- A direct question to the viewer
- A striking statistic or prophetic statement
- A counter-intuitive claim that demands resolution

## Prompt Engineering Notes

The current `DOMAIN_CONTEXT` in `llm_scorer.py` covers the basics.
Consider adding:
- Examples of complete vs. incomplete rulings (few-shot)
- Explicit instruction to not penalise Islamic vocabulary as "jargon"
- Instruction to weight emotional resonance higher for du'a (supplication) segments

## Speaker Identification
Khutbas often include:
- An imam or khateeb (primary speaker)
- Occasional congregational responses ("Ameen")
- Recorded audio that the imam plays

Congregational responses should be excluded from clips.
If speaker diarization is added (v2), filter out non-primary-speaker segments.
