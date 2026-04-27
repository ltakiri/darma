"""
drama_script_parser.py
────────────────────────────────────────────────────────────────────────────────
Parse a "drama script" text into a list of voice-segments, each carrying
the speaker identity, optional style instruction, the line text, and the
generation mode (custom_voice | voice_design | voice_clone).

Supported markup syntax
────────────────────────
1. Named predefined speaker
   [Alice]: Good morning everyone.

2. Named speaker + style override
   [Bob|excited]: I can't believe it!

3. Anonymous voice-design (no predefined speaker)
   [voice: a deep, gravelly old man]: Darkness fell over the land.

4. Continuation line (no tag → previous speaker is reused)
   Hello world.          ← reuses last speaker

5. Blank lines / comment lines (starts with #) are silently skipped.

Returned structure
──────────────────
Each element in the returned list is a dict:

{
    "speaker"   : str | None,        # predefined speaker name (lowercase) OR None
    "instruct"  : str | None,        # style instruction text OR None
    "text"      : str,               # the line to synthesise
    "mode"      : "custom_voice"     # → predefined speaker
                | "voice_design"     # → free-form voice description
                | "voice_clone",     # → clone from ref audio (populated later)
    "ref_audio" : tuple | None,      # (np.ndarray, sr) filled by UI layer
    "ref_text"  : str | None,        # transcript for ICL, filled by UI layer
}
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

# ── regex patterns ────────────────────────────────────────────────────────────

# [SpeakerName]  or  [SpeakerName|style instruction]
_RE_NAMED_TAG = re.compile(
    r"^\[([^\]|]+?)(?:\|([^\]]*))?\]\s*[:：]?\s*",
    re.IGNORECASE,
)

# [voice: free-form description]
_RE_VOICE_TAG = re.compile(
    r"^\[voice\s*:\s*([^\]]+)\]\s*[:：]?\s*",
    re.IGNORECASE,
)

# [clone: speaker_label]  – resolved against a caller-supplied clone registry
_RE_CLONE_TAG = re.compile(
    r"^\[clone\s*:\s*([^\]]+)\]\s*[:：]?\s*",
    re.IGNORECASE,
)

# ── public API ────────────────────────────────────────────────────────────────

def parse_script(
    script_text: str,
    known_speakers: Optional[set] = None,
    default_speaker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse *script_text* into a list of voice-segment dicts.

    Parameters
    ----------
    script_text
        Raw drama script text with optional speaker tags.
    known_speakers
        Set of lower-cased predefined speaker names that the TTS model
        supports.  When a tag name matches one of these the segment is
        rendered in ``custom_voice`` mode.  Unknown names whose tag contains
        no style instruction will also be treated as ``custom_voice`` with the
        raw name forwarded — the TTS wrapper will raise if it is truly unknown.
    default_speaker
        Speaker used for untagged continuation lines.  When *None* the first
        encountered speaker is used.
    """
    if known_speakers is None:
        known_speakers = set()
    else:
        known_speakers = {s.lower() for s in known_speakers}

    segments: List[Dict[str, Any]] = []
    last_speaker: Optional[str] = default_speaker
    last_instruct: Optional[str] = None
    last_mode: str = "custom_voice"

    for raw_line in script_text.splitlines():
        line = raw_line.strip()

        # skip blanks and comments
        if not line or line.startswith("#"):
            continue

        # ── [clone: label] ────────────────────────────────────────────────
        m_clone = _RE_CLONE_TAG.match(line)
        if m_clone:
            label = m_clone.group(1).strip()
            text = line[m_clone.end():].strip()
            if not text:
                continue
            seg = _make_segment(
                speaker=label,
                instruct=None,
                text=text,
                mode="voice_clone",
            )
            last_speaker = label
            last_instruct = None
            last_mode = "voice_clone"
            segments.append(seg)
            continue

        # ── [voice: description] ─────────────────────────────────────────
        m_voice = _RE_VOICE_TAG.match(line)
        if m_voice:
            description = m_voice.group(1).strip()
            text = line[m_voice.end():].strip()
            if not text:
                continue
            seg = _make_segment(
                speaker=None,
                instruct=description,
                text=text,
                mode="voice_design",
            )
            last_speaker = None
            last_instruct = description
            last_mode = "voice_design"
            segments.append(seg)
            continue

        # ── [SpeakerName] or [SpeakerName|style] ─────────────────────────
        m_named = _RE_NAMED_TAG.match(line)
        if m_named:
            raw_name = m_named.group(1).strip()
            style = (m_named.group(2) or "").strip() or None
            text = line[m_named.end():].strip()
            if not text:
                continue

            name_lc = raw_name.lower()
            if name_lc in known_speakers:
                mode = "custom_voice"
            elif style:
                # has a style hint but name is unknown → voice_design
                mode = "voice_design"
            else:
                # unknown name, no style → still forward as custom_voice
                # (OVQwen3TTSModel will raise a clear error if truly unsupported)
                mode = "custom_voice"

            seg = _make_segment(
                speaker=raw_name if mode == "custom_voice" else None,
                instruct=style,
                text=text,
                mode=mode,
            )
            last_speaker = raw_name if mode == "custom_voice" else None
            last_instruct = style
            last_mode = mode
            segments.append(seg)
            continue

        # ── untagged line → continuation ──────────────────────────────────
        text = line.strip()
        if not text:
            continue

        seg = _make_segment(
            speaker=last_speaker,
            instruct=last_instruct,
            text=text,
            mode=last_mode,
        )
        segments.append(seg)

    return segments


def segments_summary(segments: List[Dict[str, Any]]) -> str:
    """Return a human-readable summary of parsed segments (for UI display)."""
    lines = []
    for i, seg in enumerate(segments, 1):
        if seg["mode"] == "custom_voice":
            tag = f"[{seg['speaker'] or 'unknown'}]"
            if seg["instruct"]:
                tag += f" ({seg['instruct']})"
        elif seg["mode"] == "voice_design":
            tag = f"[voice: {seg['instruct'] or '?'}]"
        else:
            tag = f"[clone: {seg['speaker'] or '?'}]"
        snippet = seg["text"][:60] + ("…" if len(seg["text"]) > 60 else "")
        lines.append(f"  {i:>3}. {tag}  →  {snippet}")
    return "\n".join(lines)


def merge_audio_segments(
    wavs: List[Any],          # list of np.ndarray
    sample_rate: int,
    silence_ms: int = 300,
) -> Any:
    """
    Concatenate a list of mono float32 waveforms with short silences between them.

    Parameters
    ----------
    wavs
        Ordered list of numpy float32 arrays (one per segment).
    sample_rate
        Common sample rate of all wavs.
    silence_ms
        Milliseconds of silence to insert between consecutive segments.

    Returns
    -------
    np.ndarray
        Single concatenated waveform.
    """
    import numpy as np

    if not wavs:
        return np.zeros(0, dtype=np.float32)

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    pieces = []
    for i, wav in enumerate(wavs):
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=-1).astype(np.float32)
        pieces.append(arr)
        if i < len(wavs) - 1:
            pieces.append(silence)

    return np.concatenate(pieces)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_segment(
    speaker: Optional[str],
    instruct: Optional[str],
    text: str,
    mode: str,
) -> Dict[str, Any]:
    return {
        "speaker":   speaker,
        "instruct":  instruct,
        "text":      text,
        "mode":      mode,
        "ref_audio": None,
        "ref_text":  None,
    }
