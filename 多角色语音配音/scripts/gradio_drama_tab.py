"""
gradio_drama_tab.py
────────────────────────────────────────────────────────────────────────────────
Drop-in Gradio tab for multi-role voice acting ("Audio Drama") mode.

Usage (add to your existing gradio_helper.py make_demo, or launch standalone):

    from gradio_drama_tab import add_drama_tab

    with gr.Blocks(...) as demo:
        # ... your existing tabs ...
        add_drama_tab(demo, ov_model)          # CustomVoice model
        # or
        add_drama_tab(demo, ov_model, model_type="base")   # Base (voice-clone) model

The tab generates a single audio file that seamlessly stitches multiple
speaker voices together, one voice segment at a time, then concatenates
them with configurable silence gaps.

Script syntax (shown in the UI)
────────────────────────────────
[Alice]: Good morning.
[Bob|excited]: Did you hear the news?
[voice: a deep whispering narrator]: And so the story began.
[clone: CustomRef]: I sound like the uploaded reference voice.
Hello again.                     ← continues with last speaker
"""

from __future__ import annotations

import io
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gradio as gr
from scipy.io.wavfile import write as wav_write

from drama_script_parser import (
    parse_script,
    segments_summary,
    merge_audio_segments,
)

# ── speaker list (mirrors gradio_helper.py) ───────────────────────────────────
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna",
    "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian",
]
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian",
]

# ── default example script ────────────────────────────────────────────────────
EXAMPLE_SCRIPT = """\
# Example audio drama — edit freely
[Vivian]: Welcome to the OpenVINO audio drama showcase.
[Ryan|cheerful]: I'm Ryan, and I'm absolutely thrilled to be here!
[Eric]: Let me add a calm, measured perspective to this conversation.
[voice: a mysterious ancient sage with a slow cadence]: The secrets of the universe are not easily revealed.
[Vivian]: Well said. Shall we continue?
[Ryan]: Absolutely — this is going to be a great show!
"""

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
.drama-script textarea { font-family: 'Fira Code', 'Cascadia Code', monospace !important; font-size: 13px !important; }
.drama-preview        { font-family: 'Fira Code', 'Cascadia Code', monospace !important; font-size: 12px !important;
                        background: #1e1e2e !important; color: #cdd6f4 !important; border-radius: 8px !important; padding: 10px !important; }
.speaker-row          { border-left: 3px solid #89b4fa; padding-left: 10px; margin-bottom: 6px; }
"""

# ── clone registry helpers ────────────────────────────────────────────────────

def _build_clone_registry(
    clone_rows: List[Tuple[str, Any, str]],
) -> Dict[str, Dict]:
    """
    *clone_rows* is a list of (label, audio_tuple, ref_text) as returned by
    the Gradio clone-audio widgets.  Returns a dict keyed by lower-cased label.
    """
    registry: Dict[str, Dict] = {}
    for label, audio, ref_text in clone_rows:
        label = (label or "").strip()
        if not label:
            continue
        registry[label.lower()] = {
            "audio": audio,     # (sr, np.ndarray) from gr.Audio
            "ref_text": (ref_text or "").strip(),
        }
    return registry


def _audio_tuple_to_np(audio_tuple) -> Optional[Tuple[np.ndarray, int]]:
    """Convert Gradio (sr, wav) → (float32_wav, sr) normalised tuple."""
    if audio_tuple is None:
        return None
    sr, wav = audio_tuple
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    # normalise amplitude to [-1, 1]
    peak = np.max(np.abs(wav))
    if peak > 1.0 + 1e-6:
        wav = wav / peak
    return wav, int(sr)


# ── core generation function ──────────────────────────────────────────────────

def generate_drama(
    ov_model,
    model_type: str,
    script_text: str,
    language: str,
    silence_ms: int,
    max_new_tokens: int,
    # clone inputs (up to 4 clone slots)
    clone_label_0: str, clone_audio_0, clone_reftext_0: str,
    clone_label_1: str, clone_audio_1, clone_reftext_1: str,
    clone_label_2: str, clone_audio_2, clone_reftext_2: str,
    clone_label_3: str, clone_audio_3, clone_reftext_3: str,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[Any], str]:
    """Generate a full multi-role audio drama from a marked-up script."""

    if not script_text or not script_text.strip():
        return None, "❌ Script is empty."

    # Build clone registry
    clone_rows = [
        (clone_label_0, clone_audio_0, clone_reftext_0),
        (clone_label_1, clone_audio_1, clone_reftext_1),
        (clone_label_2, clone_audio_2, clone_reftext_2),
        (clone_label_3, clone_audio_3, clone_reftext_3),
    ]
    clone_registry = _build_clone_registry(clone_rows)

    # Parse script
    known = set(s.lower() for s in SPEAKERS)
    try:
        segments = parse_script(script_text, known_speakers=known)
    except Exception as exc:
        return None, f"❌ Script parse error: {exc}"

    if not segments:
        return None, "❌ No speakable lines found in script."

    # Attach clone audio to segments that need it
    for seg in segments:
        if seg["mode"] == "voice_clone":
            label = (seg["speaker"] or "").lower()
            if label in clone_registry:
                entry = clone_registry[label]
                tup = _audio_tuple_to_np(entry["audio"])
                seg["ref_audio"] = tup
                seg["ref_text"] = entry["ref_text"] or None
            else:
                # No clone audio supplied → degrade to custom_voice if speaker name known
                spk = (seg["speaker"] or "").lower()
                if spk in known:
                    seg["mode"] = "custom_voice"
                else:
                    return None, (
                        f"❌ [clone: {seg['speaker']}] referenced but no audio uploaded "
                        f"in the Clone Voices panel for that label."
                    )

    total = len(segments)
    wavs: List[np.ndarray] = []
    sample_rate: int = 24000
    status_lines: List[str] = []
    t_start = time.time()

    progress(0, desc="Starting drama generation…")

    for idx, seg in enumerate(segments):
        label_str = (
            f"[{seg['speaker']}]" if seg["mode"] == "custom_voice"
            else f"[clone: {seg['speaker']}]" if seg["mode"] == "voice_clone"
            else f"[voice: {seg['instruct'][:30]}…]"
        )
        progress(idx / total, desc=f"Segment {idx+1}/{total} — {label_str}")

        try:
            if seg["mode"] == "custom_voice":
                w_list, sr = ov_model.generate_custom_voice(
                    text=seg["text"],
                    language=language if language != "Auto" else None,
                    speaker=(seg["speaker"] or "Ryan").lower().replace(" ", "_"),
                    instruct=seg["instruct"] or None,
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
                )

            elif seg["mode"] == "voice_design":
                if model_type not in ("custom_voice", "voice_design"):
                    # voice_design is only available on the voice_design model type;
                    # fall back to first available speaker on custom_voice model
                    w_list, sr = ov_model.generate_custom_voice(
                        text=seg["text"],
                        language=language if language != "Auto" else None,
                        speaker="vivian",
                        instruct=seg["instruct"] or None,
                        non_streaming_mode=True,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    w_list, sr = ov_model.generate_voice_design(
                        text=seg["text"],
                        language=language if language != "Auto" else None,
                        instruct=seg["instruct"] or "Speak naturally.",
                        non_streaming_mode=True,
                        max_new_tokens=max_new_tokens,
                    )

            elif seg["mode"] == "voice_clone":
                ref_audio_tuple = seg["ref_audio"]
                w_list, sr = ov_model.generate_voice_clone(
                    text=seg["text"],
                    language=language if language != "Auto" else None,
                    ref_audio=ref_audio_tuple,
                    ref_text=seg["ref_text"] or None,
                    x_vector_only_mode=(seg["ref_text"] is None or seg["ref_text"] == ""),
                    non_streaming_mode=True,
                    max_new_tokens=max_new_tokens,
                )
            else:
                continue

            wav = np.asarray(w_list[0], dtype=np.float32)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=-1).astype(np.float32)
            wavs.append(wav)
            sample_rate = int(sr)
            status_lines.append(f"  ✓ {idx+1:>3}/{total}  {label_str}  ({len(wav)/sr:.1f}s)")

        except Exception as exc:
            status_lines.append(f"  ✗ {idx+1:>3}/{total}  {label_str}  ERROR: {exc}")
            # insert silence so timing stays approximately correct
            wavs.append(np.zeros(int(sample_rate * 0.5), dtype=np.float32))

    if not wavs:
        return None, "❌ All segments failed."

    progress(1.0, desc="Concatenating audio…")
    combined = merge_audio_segments(wavs, sample_rate, silence_ms=silence_ms)

    elapsed = time.time() - t_start
    total_dur = len(combined) / sample_rate
    rtf = elapsed / max(total_dur, 0.01)

    summary = (
        f"✓ Drama generated: {total_dur:.1f}s audio from {total} segments\n"
        f"  Inference: {elapsed:.1f}s  |  RTF: {rtf:.3f}\n\n"
        "Segment log:\n" + "\n".join(status_lines)
    )

    return (sample_rate, combined), summary


# ── preview helper (no model needed) ─────────────────────────────────────────

def preview_script(script_text: str) -> str:
    """Parse and return a formatted segment summary without generating audio."""
    if not script_text or not script_text.strip():
        return "No script entered."
    known = set(s.lower() for s in SPEAKERS)
    try:
        segs = parse_script(script_text, known_speakers=known)
    except Exception as exc:
        return f"Parse error: {exc}"
    if not segs:
        return "No speakable lines found."
    return f"Found {len(segs)} segment(s):\n\n" + segments_summary(segs)


# ── public function to add tab ────────────────────────────────────────────────

def add_drama_tab(
    parent_blocks,
    ov_model,
    model_type: str = "custom_voice",
    tab_label: str = "🎭 Audio Drama",
) -> None:
    """
    Append an Audio Drama tab to an existing ``gr.Blocks`` context.

    Parameters
    ----------
    parent_blocks
        The ``gr.Blocks`` object (must be called inside a ``with`` block or
        passed after construction but before ``.launch()``).
    ov_model
        An ``OVQwen3TTSModel`` instance.
    model_type
        One of ``"custom_voice"``, ``"base"``, ``"voice_design"``.
    tab_label
        Label shown on the Gradio tab.
    """
    with gr.Tab(tab_label):
        gr.Markdown(
            """
## 🎭 Multi-Role Audio Drama

Write a script with **speaker tags** — each line is synthesised with a
different voice and stitched into one seamless audio file.

| Tag syntax | Effect |
|---|---|
| `[Alice]:` text | Predefined speaker (CustomVoice model) |
| `[Bob\|excited]:` text | Predefined speaker + style instruction |
| `[voice: description]:` text | Free-form voice design |
| `[clone: label]:` text | Clone from uploaded reference audio |
| Untagged line | Continues with the previous speaker |
"""
        )

        with gr.Row():
            # ── left column: script + options ─────────────────────────
            with gr.Column(scale=3):
                script_box = gr.Textbox(
                    label="Drama Script",
                    lines=18,
                    placeholder=EXAMPLE_SCRIPT,
                    value=EXAMPLE_SCRIPT,
                    elem_classes=["drama-script"],
                )

                with gr.Row():
                    lang_dd = gr.Dropdown(
                        label="Language (applied to all segments)",
                        choices=LANGUAGES,
                        value="Auto",
                        interactive=True,
                        scale=2,
                    )
                    silence_sl = gr.Slider(
                        label="Silence between lines (ms)",
                        minimum=0, maximum=2000, value=300, step=50,
                        scale=2,
                    )
                    tokens_sl = gr.Slider(
                        label="Max new tokens per segment",
                        minimum=256, maximum=4096, value=1024, step=128,
                        scale=2,
                    )

                with gr.Row():
                    preview_btn = gr.Button("👁 Preview Segments", variant="secondary")
                    generate_btn = gr.Button("🎬 Generate Drama", variant="primary")

                preview_box = gr.Textbox(
                    label="Segment Preview",
                    lines=8,
                    interactive=False,
                    elem_classes=["drama-preview"],
                )

            # ── right column: output + clone voices ───────────────────
            with gr.Column(scale=2):
                audio_out = gr.Audio(
                    label="Generated Drama Audio",
                    type="numpy",
                )
                status_box = gr.Textbox(
                    label="Generation Log",
                    lines=10,
                    interactive=False,
                )

                # Clone voice panel (only when base model or always available)
                with gr.Accordion("🎤 Clone Voices (optional)", open=False):
                    gr.Markdown(
                        "Upload reference audio clips and give each a **label** "
                        "matching `[clone: label]` tags in your script.  "
                        "Reference text improves quality (ICL mode); leave blank "
                        "to use x-vector-only mode."
                    )
                    clone_components = []
                    for slot_i in range(4):
                        with gr.Group(elem_classes=["speaker-row"]):
                            with gr.Row():
                                lbl = gr.Textbox(
                                    label=f"Clone label {slot_i+1}",
                                    placeholder=f"e.g. CustomRef{slot_i+1}",
                                    scale=1,
                                )
                                aud = gr.Audio(
                                    label=f"Reference audio {slot_i+1}",
                                    type="numpy",
                                    scale=2,
                                )
                            ref_txt = gr.Textbox(
                                label=f"Reference text {slot_i+1} (transcript of the audio above)",
                                placeholder="Enter the exact words spoken in the reference audio…",
                                lines=2,
                            )
                        clone_components.extend([lbl, aud, ref_txt])

        # ── wire up events ────────────────────────────────────────────
        preview_btn.click(
            fn=preview_script,
            inputs=[script_box],
            outputs=[preview_box],
        )

        # Flatten clone components for the generate function
        (
            cl0, ca0, cr0,
            cl1, ca1, cr1,
            cl2, ca2, cr2,
            cl3, ca3, cr3,
        ) = clone_components

        generate_btn.click(
            fn=lambda *args, **kw: generate_drama(ov_model, model_type, *args, **kw),
            inputs=[
                script_box, lang_dd, silence_sl, tokens_sl,
                cl0, ca0, cr0,
                cl1, ca1, cr1,
                cl2, ca2, cr2,
                cl3, ca3, cr3,
            ],
            outputs=[audio_out, status_box],
        )


# ── standalone launcher (for quick testing without existing demo) ─────────────

def make_drama_demo(ov_model, model_type: str = "custom_voice") -> gr.Blocks:
    """
    Create a standalone Gradio demo with only the Audio Drama tab.

    Useful for testing or if you want a dedicated drama-only interface.
    """
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=_CSS, title="Qwen3-TTS Audio Drama") as demo:
        gr.Markdown("# 🎭 Qwen3-TTS Audio Drama (OpenVINO)")
        add_drama_tab(demo, ov_model, model_type=model_type, tab_label="Audio Drama")

    return demo
