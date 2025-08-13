import gradio as gr
import srt
import torch
import torchaudio
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import warnings
import re
from num2words import num2words
from TTS.api import TTS
from datetime import timedelta, datetime
import pooch
import tempfile
import whisper
import json
import shutil
import traceback
import sys
import gc
import time
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which as pydub_which
import pysrt
from collections import namedtuple
from loguru import logger


# Stable-TS Imports
import stable_whisper
import stable_whisper.audio

# Local Backend Import for Fine-Tuning
try:
    import Coqui_XTTSv2_train_module as train_module
except ImportError:
    logger.critical("WARNING: Coqui_XTTSv2_train_module.py not found. The Fine-Tuning tab will be disabled.")
    train_module = None


# Higgs module import
try:
    import higgs_v3_module as higgs
except ImportError:
    logger.critical("WARNING: higgs_v3_module.py not found. The Higgs tab will be disabled.")
    higgs = None

# Coqui XTTSv2 Import
try:
    import Coqui_XTTSv2_module as coqui_xtts
except ImportError:
    logger.critical("WARNING: Coqui_XTTSv2_module.py not found. The Coqui XTTSv2 tab will be disabled.")
    coqui_xtts = None

# ========================================================================================
# --- Global Model and Configuration ---
# ========================================================================================

WHISPER_CONFIG_LIBRARY_PATH = "whisper_configs"

# Global variables to hold the models
whisper_model = None
stable_whisper_model = None 
current_whisper_device = None
current_whisper_model_size = None
current_whisper_engine = None


# --- Device Auto-Detection ---
def get_available_devices():
    """Checks for CUDA availability and returns a list of devices."""
    if torch.cuda.is_available():
        return ["cuda", "cpu"]
    return ["cpu"]

AVAILABLE_DEVICES = get_available_devices()


# ========================================================================================
# --- System Prerequisite Check ---
# ========================================================================================
def check_ffmpeg():
    """Checks if FFmpeg is installed and in the system's PATH."""
    if shutil.which("ffmpeg") or pydub_which("ffmpeg"):
        logger.success("✅ FFmpeg found.")
        return True
    else:
        logger.critical("❌ FFmpeg not found.")
        logger.info("This application requires FFmpeg for audio processing.")
        logger.info("Please install it and ensure it's in your system's PATH.")
        logger.info("Installation instructions:")
        logger.info("  - Windows: Download from https://ffmpeg.org/download.html (and add to PATH)")
        logger.info("  - MacOS (via Homebrew): brew install ffmpeg")
        logger.info("  - Linux (Debian/Ubuntu): sudo apt update && sudo apt install ffmpeg")
        return False

# ========================================================================================
# --- Library and Config Functions ---
# ========================================================================================

os.makedirs(WHISPER_CONFIG_LIBRARY_PATH, exist_ok=True)
os.makedirs("gradio_outputs", exist_ok=True)
os.makedirs("whisper_outputs", exist_ok=True)

    
def get_whisper_config_files():
    if not os.path.exists(WHISPER_CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(WHISPER_CONFIG_LIBRARY_PATH) if f.endswith('.json')]

def save_whisper_config(*args):
    keys = [
        "config_name", "model_size", "language", "task", "output_action", "whisper_engine", "autorun", "tts_config",
        "regroup_enabled", "regroup_string", "suppress_silence", "vad_enabled", "vad_threshold",
        "min_word_dur", "no_speech_threshold", "logprob_threshold", "compression_ratio_threshold",
        "temperature", "condition_on_previous_text", "initial_prompt", "demucs_enabled", "only_voice_freq",
        "suppress_ts_tokens", "time_scale",
        "post_processing_enabled", "refine_enabled", "refine_steps", "refine_precision",
        "remove_repetitions_enabled", "remove_repetitions_max_words",
        "remove_words_str_enabled", "words_to_remove", "find_replace_enabled", "find_word", "replace_word",
        "split_by_gap_enabled", "split_by_gap_value",
        "split_by_punctuation_enabled", "split_by_length_enabled", "split_by_length_max_chars", "split_by_length_max_words",
        "split_by_duration_enabled", "split_by_duration_max_dur",
        "merge_by_gap_enabled", "merge_by_gap_min_gap", "merge_by_gap_max_words",
        "merge_by_punctuation_enabled", "merge_by_punctuation_string",
        "merge_all_segments_enabled",
        "fill_gaps_enabled", "fill_gaps_min_gap"
    ]
    config_data = dict(zip(keys, args))
    config_name = config_data.pop("config_name")

    if not config_name or not config_name.strip():
        return "❌ Error: Please enter a name for the configuration."
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", config_name).strip().replace(" ", "_")
    if not sanitized_name: return "❌ Error: Invalid config name."

    config_path = os.path.join(WHISPER_CONFIG_LIBRARY_PATH, f"{sanitized_name}.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=4)
        return f"✅ Config '{sanitized_name}' saved successfully."
    except Exception as e:
        return f"❌ Error saving config: {e}"

def load_whisper_config(config_name):
    keys = [
        "model_size", "language", "task", "output_action", "whisper_engine", "autorun", "tts_config",
        "regroup_enabled", "regroup_string", "suppress_silence", "vad_enabled", "vad_threshold",
        "min_word_dur", "no_speech_threshold", "logprob_threshold", "compression_ratio_threshold",
        "temperature", "condition_on_previous_text", "initial_prompt", "demucs_enabled", "only_voice_freq",
        "suppress_ts_tokens", "time_scale",
        "post_processing_enabled", "refine_enabled", "refine_steps", "refine_precision",
        "remove_repetitions_enabled", "remove_repetitions_max_words",
        "remove_words_str_enabled", "words_to_remove", "find_replace_enabled", "find_word", "replace_word",
        "split_by_gap_enabled", "split_by_gap_value",
        "split_by_punctuation_enabled", "split_by_length_enabled", "split_by_length_max_chars", "split_by_length_max_words",
        "split_by_duration_enabled", "split_by_duration_max_dur",
        "merge_by_gap_enabled", "merge_by_gap_min_gap", "merge_by_gap_max_words",
        "merge_by_punctuation_enabled", "merge_by_punctuation_string",
        "merge_all_segments_enabled",
        "fill_gaps_enabled", "fill_gaps_min_gap"
    ]
    if not config_name: return [gr.update()] * (len(keys) + 2)
    config_path = os.path.join(WHISPER_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return [gr.update()] * (len(keys) + 2)
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
        
        updates = [gr.update(value=config_data.get(k)) for k in keys]
        
        stable_ts_options_visible = config_data.get("whisper_engine") == "Stable-TS"
        updates.append(gr.update(visible=stable_ts_options_visible))

        post_processing_accordion_visible = config_data.get("post_processing_enabled", False)
        updates.append(gr.update(visible=post_processing_accordion_visible))
        
        return updates
    except Exception as e:
        logger.critical(f"Error loading Whisper config {config_name}: {e}")
        return [gr.update()] * (len(keys) + 2)

def delete_whisper_config(config_name):
    if not config_name: return "ℹ️ No config selected to delete."
    config_path = os.path.join(WHISPER_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return f"❌ Error: Config '{config_name}' not found."
    try:
        os.remove(config_path)
        return f"✅ Config '{config_name}' deleted successfully."
    except Exception as e:
        return f"❌ Error deleting config: {e}"


# ========================================================================================
# --- Model Loading Functions ---
# ========================================================================================

def load_whisper_model(model_size, device, engine):
    global whisper_model, stable_whisper_model, current_whisper_device, current_whisper_model_size, current_whisper_engine
    if current_whisper_model_size == model_size and current_whisper_device == device and current_whisper_engine == engine:
        return "Whisper model is already loaded."
    
    logger.info(f"⏳ Loading {engine} model '{model_size}' to device: {device}...")
    if engine == "OpenAI Whisper":
        if stable_whisper_model is not None: stable_whisper_model = None; gc.collect(); torch.cuda.empty_cache()
        whisper_model = whisper.load_model(model_size, device=device)
    elif engine == "Stable-TS":
        if whisper_model is not None: whisper_model = None; gc.collect(); torch.cuda.empty_cache()
        stable_whisper_model = stable_whisper.load_model(model_size, device=device)
    
    current_whisper_device = device
    current_whisper_model_size = model_size
    current_whisper_engine = engine
    logger.success(f"✅ {engine} Model loaded successfully on {device}.")
    return "Whisper model is ready."


# ========================================================================================
# --- Gradio Processing Functions ---
# ========================================================================================
def run_whisper_transcription(
    audio_file_path, model_size, language, task, output_action, whisper_device, whisper_engine,
    regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
    min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
    temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
    suppress_ts_tokens, time_scale,
    post_processing_enabled, refine_enabled, refine_steps, refine_precision,
    remove_repetitions_enabled, remove_repetitions_max_words,
    remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
    split_by_gap_enabled, split_by_gap_value,
    split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
    split_by_duration_enabled, split_by_duration_max_dur,
    merge_by_gap_enabled, merge_by_gap_min_gap, merge_by_gap_max_words,
    merge_by_punctuation_enabled, merge_by_punctuation_string,
    merge_all_segments_enabled,
    fill_gaps_enabled, fill_gaps_file_input, fill_gaps_min_gap,
    progress=gr.Progress(track_tqdm=True)
):
    if audio_file_path is None: return "❌ Error: Please upload an audio file.", "", [], None
    try:
        progress(0, desc=f"Loading {whisper_engine} Model...")
        load_whisper_model(model_size, whisper_device, whisper_engine)
        lang = language if language and language.strip() else None
        
        progress(0.1, desc="Loading audio file...")
        audio_array = coqui_xtts.safe_load_audio(audio_file_path)

        progress(0.2, desc=f"Starting {whisper_engine} {task}...")
        
        if whisper_engine == "OpenAI Whisper":
            result_dict = whisper_model.transcribe(audio_array, language=lang, task=task, verbose=True)
            full_text = result_dict['text']
            Segment = namedtuple('Segment', ['start', 'end', 'text'])
            segments = [Segment(s['start'], s['end'], s['text']) for s in result_dict['segments']]

        elif whisper_engine == "Stable-TS":
            regroup_param = regroup_string if regroup_string and regroup_string.strip() else regroup_enabled
            result_obj = stable_whisper_model.transcribe(
                audio_array, language=lang, task=task, verbose=True, regroup=regroup_param,
                suppress_silence=suppress_silence, vad=vad_enabled, vad_threshold=vad_threshold,
                min_word_dur=min_word_dur, no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold, compression_ratio_threshold=compression_ratio_threshold,
                temperature=temperature, condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt if initial_prompt and initial_prompt.strip() else None,
                demucs=demucs_enabled, only_voice_freq=only_voice_freq,
                suppress_ts_tokens=suppress_ts_tokens, time_scale=time_scale
            )
            
            if post_processing_enabled:
                progress(0.8, desc="Applying Post-Processing...")
                
                # Splitting
                if split_by_gap_enabled: result_obj.split_by_gap(split_by_gap_value)
                if split_by_punctuation_enabled: result_obj.split_by_punctuation(punctuation='.?!,')
                if split_by_length_enabled: result_obj.split_by_length(split_by_length_max_chars, split_by_length_max_words)
                if split_by_duration_enabled: result_obj.split_by_duration(split_by_duration_max_dur)

                # Merging
                if merge_by_gap_enabled:
                    logger.info("Applying Merge by Gap...")
                    result_obj.merge_by_gap(min_gap=merge_by_gap_min_gap, max_words=merge_by_gap_max_words)
                if merge_by_punctuation_enabled and merge_by_punctuation_string:
                    logger.info("Applying Merge by Punctuation...")
                    punctuation_list = [p.strip() for p in merge_by_punctuation_string.split(',') if p.strip()]
                    result_obj.merge_by_punctuation(punctuation=punctuation_list)
                if merge_all_segments_enabled:
                    logger.info("Merging all segments...")
                    result_obj.merge_all_segments()

                # Refinement & Cleaning
                if refine_enabled: stable_whisper_model.refine(audio_array, result_obj, steps=refine_steps, precision=refine_precision, verbose=False)
                if remove_repetitions_enabled: result_obj.remove_repetition(max_words=remove_repetitions_max_words, verbose=False)
                if remove_words_str_enabled and words_to_remove:
                    result_obj.remove_words_by_str([w.strip() for w in words_to_remove.split(',') if w.strip()], verbose=False)
                if find_replace_enabled and find_word and replace_word is not None:
                    result_obj = coqui_xtts.find_and_replace(result_obj, find_word, replace_word)

                # Filling Gaps (last)
                if fill_gaps_enabled and fill_gaps_file_input is not None:
                    logger.info(f"Applying Fill Gaps using file: {fill_gaps_file_input.name}")
                    try:
                        result_obj.fill_in_gaps(fill_gaps_file_input.name, min_gap=fill_gaps_min_gap)
                    except Exception as e:
                        logger.warning(f"⚠️ Warning: Could not fill gaps. Error: {e}")

            full_text = result_obj.text
            segments = result_obj.segments
        
        if output_action == "Display Only": 
            return full_text, "", [], None

        output_dir, base_name, timestamp = "whisper_outputs", os.path.splitext(os.path.basename(audio_file_path))[0], datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        txt_content = full_text
        srt_segments = [srt.Subtitle(i+1, timedelta(seconds=seg.start), timedelta(seconds=seg.end), seg.text.strip()) for i, seg in enumerate(segments)]
        srt_content = srt.compose(srt_segments)
        vtt_content = "WEBVTT\n\n" + "\n\n".join(f"{timedelta(seconds=seg.start)} --> {timedelta(seconds=seg.end)}\n{seg.text.strip()}" for seg in segments)
        json_content = json.dumps({"text": full_text, "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]}, indent=2, ensure_ascii=False)

        if output_action == "Save All Formats (.txt, .srt, .vtt, .json)":
            file_paths = []
            for ext, content in [("txt", txt_content), ("srt", srt_content), ("vtt", vtt_content), ("json", json_content)]:
                filepath = os.path.join(output_dir, f"{base_name}_{timestamp}.{ext}")
                with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
                file_paths.append(filepath)
            return full_text, srt_content, file_paths, None

        elif "Pipeline" in output_action:
            ext = "txt" if output_action == "Pipeline .txt to TTS" else "srt"
            content = txt_content if ext == "txt" else srt_content
            pipeline_filepath = os.path.join(output_dir, f"{base_name}_{timestamp}_pipelined.{ext}")
            with open(pipeline_filepath, 'w', encoding='utf-8') as f: f.write(content)
            return full_text, content, [pipeline_filepath], pipeline_filepath

    except Exception as e:
        logger.critical("\n--- DETAILED WHISPER ERROR ---"); traceback.print_exc(); print("------------------------------\n")
        return f"❌ An unexpected error occurred: {e}", "", [], None


def _resolve_speaker_ref(uploaded_path, textbox_path, output_base_path):
    # 1) Prefer the newly uploaded file
    if uploaded_path and os.path.exists(uploaded_path):
        return uploaded_path
    # 2) Fall back to a manual path typed/pasted
    if textbox_path and os.path.exists(textbox_path):
        return textbox_path
    # 3) Last resort: try to auto-discover from latest run
    try:
        if output_base_path and train_module is not None:
            c, v, ckpt, spk, _ = train_module.autofill_ft_paths(output_base_path)
            if spk and os.path.exists(spk):
                return spk
    except Exception:
        pass
    return ""  # let backend error gracefully if still empty

def _run_ft_inference(ui_text_entry, txt_file_input, srt_vtt_file_input, input_mode, srt_timing_mode, tts_language, speaker_reference_upload):
    return train_module.ft_inference_generate(
        ui_text_entry,
        txt_file_input,
        srt_vtt_file_input,
        input_mode,
        srt_timing_mode,
        tts_language,
        speaker_reference_upload
    )
    return train_module.run_tts(lang, tts_text, ref)

# ========================================================================================
# --- Gradio UI ---
# ========================================================================================
def create_gradio_ui():
    with gr.Blocks(title="Vox: The All-in-One ASR&TTS AI Suite") as demo:
        gr.HTML("""<div style="text-align: center; max-width: 800px; margin: 0 auto;"><h1 style="color: #4CAF50;">Vox: The All-in-One ASR&TTS AI Suite</h1><p style="font-size: 1.1em;">A complete toolkit for audio transcription and voiceover generation with Coqui XTTS-v2 and Higgs-v3 TTS.</p></div>""")
        with gr.Group(visible=False):
            tts_input_file = gr.File()
            tts_output_audio = gr.Audio()
            tts_status_textbox = gr.Textbox()
            tts_load_config_dd = gr.Dropdown()
            tts_refresh_configs_btn = gr.Button()
        with gr.Accordion("⚙️ Global Device & Process Settings", open=True):
            with gr.Row():
                tts_device = gr.Radio(label="TTS Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
                whisper_device = gr.Radio(label="Whisper/Higgs/Training Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
                if coqui_xtts and coqui_xtts.XTTS_AVAILABLE:
                    clear_cache_button = gr.Button("Clear Coqui TTS Cache", variant="stop")
                    cache_status = gr.Textbox(label="Cache Status", interactive=False)
        
        with gr.Tabs() as tabs:
            with gr.Tab("Whisper Transcription", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Configuration Management", open=False):
                            whisper_config_name = gr.Textbox(label="Config Name", placeholder="Enter a name to save current settings...")
                            whisper_save_config_btn = gr.Button("Save Config")
                            with gr.Row():
                                whisper_load_config_dd = gr.Dropdown(label="Load Config", choices=get_whisper_config_files(), scale=3)
                                whisper_load_config_btn = gr.Button("Load", scale=1)
                                whisper_delete_config_btn = gr.Button("Delete", variant="stop", scale=1)
                            whisper_refresh_configs_btn_main = gr.Button("Refresh Configs")
                            whisper_config_save_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("## 1. Upload Audio")
                        whisper_audio_input = gr.File(label="Input Audio", file_types=["audio"])
                        gr.Markdown("## 2. Configure Transcription")
                        
                        whisper_engine = gr.Radio(label="Transcription Engine", choices=["OpenAI Whisper", "Stable-TS"], value="OpenAI Whisper", info="Stable-TS provides more natural, sentence-based segmentation.")
                        whisper_model_size = gr.Dropdown(label="Whisper Model", choices=["tiny", "base", "small", "medium", "large", "turbo", "large-v2", "large-v3", "large-v3-turbo"], value="base")
                        whisper_language = gr.Textbox(label="Language (optional)", info="e.g., 'en', 'es'. Leave blank to auto-detect.")
                        whisper_task = gr.Radio(label="Task", choices=["transcribe", "translate"], value="transcribe")
                        whisper_info_box = gr.Markdown(visible=False)
                        
                        gr.Markdown("## 3. Choose Output Action")
                        whisper_output_action = gr.Radio(label="Action", choices=["Display Only", "Save All Formats (.txt, .srt, .vtt, .json)", "Pipeline .txt to TTS", "Pipeline .srt to TTS"], value="Display Only")
                        
                        with gr.Group(visible=False) as whisper_pipeline_group:
                            whisper_autorun_tts = gr.Checkbox(label="Auto-run TTS after pipeline", value=False)
                            whisper_tts_config = gr.Dropdown(label="TTS Config to use", choices=coqui_xtts.get_tts_config_files())
                            whisper_refresh_tts_configs_btn = gr.Button("Refresh TTS Configs")

                        with gr.Accordion("Stable-TS Advanced Options", open=False, visible=False) as stable_ts_options:
                            with gr.Tabs():
                                with gr.TabItem("Segmentation & VAD"):
                                    suppress_silence = gr.Checkbox(label="Suppress Silence", value=True)
                                    vad_enabled = gr.Checkbox(label="Use Silero VAD", value=False)
                                    vad_threshold = gr.Slider(label="VAD Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.35)
                                    min_word_dur = gr.Slider(label="Min Word Duration (s)", minimum=0.0, maximum=1.0, step=0.05, value=0.1)
                                    regroup_enabled = gr.Checkbox(label="Enable Regrouping", value=True)
                                    regroup_string = gr.Textbox(label="Custom Regrouping Algorithm", placeholder="e.g., cm_sg=.5_mg=.3+3")
                                with gr.TabItem("Confidence & Decoding"):
                                    no_speech_threshold = gr.Slider(label="No Speech Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.6)
                                    logprob_threshold = gr.Slider(label="Log Probability Threshold", minimum=-10.0, maximum=0.0, step=0.5, value=-1.0)
                                    compression_ratio_threshold = gr.Slider(label="Compression Ratio Threshold", minimum=0.0, maximum=10.0, step=0.5, value=2.4)
                                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.0)
                                    condition_on_previous_text = gr.Checkbox(label="Condition on Previous Text", value=True)
                                    initial_prompt = gr.Textbox(label="Initial Prompt")
                                with gr.TabItem("Other"):
                                    demucs_enabled = gr.Checkbox(label="Isolate Vocals (Demucs)", value=False)
                                    only_voice_freq = gr.Checkbox(label="Only Use Voice Frequencies", value=False)
                                    suppress_ts_tokens = gr.Checkbox(label="Suppress Timestamp Tokens", value=False)
                                    time_scale = gr.Slider(label="Time Scale", minimum=0.5, maximum=2.0, step=0.1, value=1.0)

                        with gr.Accordion("Post-Processing (Stable-TS Only)", open=False) as post_processing_options:
                            post_processing_enabled = gr.Checkbox(label="Enable Post-Processing", value=False)
                            with gr.Group(visible=False) as post_processing_group:
                                with gr.Tabs():
                                    with gr.TabItem("Splitting"):
                                        split_by_gap_enabled = gr.Checkbox(label="Split by Gap", value=False, info="Splits a segment wherever there is a pause or silence longer than the specified 'Max Gap'.")
                                        split_by_gap_value = gr.Slider(label="Max Gap (s)", minimum=0.0, maximum=1.0, step=0.01, value=0.1)
                                        split_by_punctuation_enabled = gr.Checkbox(label="Split by Punctuation", value=False, info="Splits segments at specified punctuation marks.")
                                        split_by_length_enabled = gr.Checkbox(label="Split by Length", value=False, info="Ensures no segment is longer than the specified max characters or words.")
                                        split_by_length_max_chars = gr.Slider(label="Max Chars", minimum=10, maximum=200, step=10, value=50)
                                        split_by_length_max_words = gr.Slider(label="Max Words", minimum=5, maximum=50, step=1, value=15)
                                        split_by_duration_enabled = gr.Checkbox(label="Split by Duration", value=False, info="Splits any segment that is longer than the specified max duration.")
                                        split_by_duration_max_dur = gr.Slider(label="Max Duration (s)", minimum=1, maximum=30, step=1, value=10)
                                    with gr.TabItem("Refinement & Cleaning"):
                                        refine_enabled = gr.Checkbox(label="Refine Timestamps", value=False, info="Improves timestamp accuracy by iteratively analyzing token probabilities.")
                                        refine_steps = gr.Radio(label="Steps", choices=["se", "s", "e"], value="se")
                                        refine_precision = gr.Slider(label="Precision (s)", minimum=0.02, maximum=1.0, step=0.01, value=0.1)
                                        remove_repetitions_enabled = gr.Checkbox(label="Remove Repetitions", value=False, info="Removes consecutively repeated words or phrases.")
                                        remove_repetitions_max_words = gr.Slider(label="Max Words in Repetition", minimum=1, maximum=10, step=1, value=1)
                                        remove_words_str_enabled = gr.Checkbox(label="Remove Specific Words", value=False, info="Removes specific words from the transcription (e.g., 'um', 'uh').")
                                        words_to_remove = gr.Textbox(label="Words to Remove (comma-separated)", placeholder="uh, um, you know")
                                    with gr.TabItem("Find & Replace"):
                                        find_replace_enabled = gr.Checkbox(label="Enable Find and Replace", value=False, info="Finds all occurrences of a word/phrase and replaces it with another.")
                                        find_word = gr.Textbox(label="Find Word/Phrase")
                                        replace_word = gr.Textbox(label="Replace With")
                                    with gr.TabItem("Merging & Filling"):
                                        gr.Markdown("### Merging Options")
                                        merge_by_gap_enabled = gr.Checkbox(label="Merge by Gap", value=False, info="Merges adjacent segments if the pause between them is smaller than the specified 'Min Gap'.")
                                        merge_by_gap_min_gap = gr.Slider(label="Min Gap (s)", minimum=0.0, maximum=1.0, step=0.05, value=0.1)
                                        merge_by_gap_max_words = gr.Slider(label="Max Words per Merged Segment", minimum=10, maximum=100, step=5, value=50)
                                        
                                        merge_by_punctuation_enabled = gr.Checkbox(label="Merge by Punctuation", value=False, info="Merges segments that have specific punctuation marks between them.")
                                        merge_by_punctuation_string = gr.Textbox(label="Punctuation to Merge On (comma-separated)", value=".?!,")

                                        merge_all_segments_enabled = gr.Checkbox(label="Merge All Segments into One", value=False, info="Merges all segments into a single, continuous segment.")
                                        
                                        gr.Markdown("### Fill Gaps")
                                        fill_gaps_enabled = gr.Checkbox(label="Fill Gaps from another Transcription", value=False, info="Uses a second transcription file to fill in any gaps in the primary transcription.")
                                        fill_gaps_file_input = gr.File(label="Reference Transcription (.json)", file_types=[".json"])
                                        fill_gaps_min_gap = gr.Slider(label="Min Gap to Fill (s)", minimum=0.0, maximum=2.0, step=0.1, value=0.1)
                        
                        transcribe_btn = gr.Button("Transcribe Audio", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("## Transcription Result")
                        whisper_output_text = gr.Textbox(label="Output Text", lines=10, interactive=True)
                        with gr.Accordion("File Content Preview & Downloads", open=False):
                            whisper_file_preview = gr.Code(label="File Preview", lines=10, interactive=False)
                            whisper_output_files = gr.File(label="Generated Files", interactive=False)
            if coqui_xtts and coqui_xtts.XTTS_AVAILABLE:
                with gr.Tab("Coqui XTTS Voiceover", id=1):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Accordion("Configuration Management", open=False):
                                tts_config_name = gr.Textbox(label="Config Name", placeholder="Enter a name to save current settings...")
                                tts_save_config_btn = gr.Button("Save Config")
                                with gr.Row():
                                    tts_load_config_dd = gr.Dropdown(label="Load Config", choices=coqui_xtts.get_tts_config_files(), scale=3)
                                    tts_load_config_btn = gr.Button("Load", scale=1)
                                    tts_delete_config_btn = gr.Button("Delete", variant="stop", scale=1)
                                tts_refresh_configs_btn = gr.Button("Refresh Configs")
                                tts_config_save_status = gr.Textbox(label="Status", interactive=False)

                            gr.Markdown("## 1. Upload Your Content")
                            tts_input_file = gr.File(label="Input File (.txt, .srt, .vtt)", file_types=['.txt', '.srt', '.vtt'])
                            tts_input_mode = gr.Radio(label="Input Mode", choices=["Text File Mode", "SRT/VTT Mode"], value="Text File Mode")
                            with gr.Group(visible=False) as tts_srt_group:
                                tts_srt_timing_mode = gr.Radio(label="SRT/VTT Timing Mode", choices=["Strict (Cut audio to fit)", "Flexible (Prevent audio cutoff)"], value="Flexible (Prevent audio cutoff)")
                            # --- Sentence Split Controls ---
                            with gr.Accordion("Split sentences", open=False):
                                tts_soft_limit = gr.Slider(label="Soft limit (characters)", minimum=120, maximum=230, step=1, value=210, info="Aim to split before this length, at punctuation if possible.")
                                tts_hard_limit = gr.Slider(label="Hard limit (characters)", minimum=200, maximum=248, step=1, value=240, info="Absolute maximum per chunk (kept ≤248 for XTTS stability).")

                            
                            gr.Markdown("## 2. Configure Voice")
                            tts_voice_mode = gr.Radio(label="Voice Mode", choices=['Clone', 'Stock'], value='Stock')
                            with gr.Group(visible=False) as tts_clone_voice_group:
                                tts_clone_source = gr.Radio(label="Clone Source", choices=["Upload New Sample", "Use from Library"], value="Upload New Sample")
                                with gr.Group(visible=True) as tts_upload_group: tts_clone_speaker_audio = gr.Audio(label="Upload Voice Sample (6-30s)", type="filepath")
                                with gr.Group(visible=False) as tts_library_group:
                                    tts_library_voice = gr.Dropdown(label="Select Library Voice", choices=coqui_xtts.get_library_voices())
                                    refresh_library_btn_tts = gr.Button("Refresh Library")
                            with gr.Group(visible=True) as tts_stock_voice_group: tts_stock_voice = gr.Dropdown(label="Stock Voice", choices=list(coqui_xtts.STOCK_VOICES.keys()), value='En')
                            
                            gr.Markdown("## 3. Configure Output")
                            tts_language = gr.Dropdown(label="Language", choices=coqui_xtts.SUPPORTED_LANGUAGES, value="en")
                            tts_output_format = gr.Radio(label="Output Format", choices=['wav', 'mp3'], value='wav')
                            
                            
                            tts_generate_btn = gr.Button("Generate Voiceover", variant="primary")

                        with gr.Column(scale=2):
                            gr.Markdown("## Generated Audio")
                            tts_output_audio = gr.Audio(label="Output", type="filepath", show_download_button=True)
                            tts_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Column():
                            gr.HTML('''
                        <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #2196f3;max-width:420px;margin-left:auto;margin-right:auto;">
                            <b>⭐ Key Features:</b><br>
                            • Cross-language support: Can clone voices across 17 different languages.<br>
                            • Emotion and style transfer: Preserves emotional characteristics and speaking style from the reference clip.<br>
                            • Multiple speaker references: v2 supports using multiple speaker references and interpolation between speakers.<br>
                            • Architectural improvements: Better speaker conditioning and stability improvements over v1.<br>
                        </div>
                        ''')
            if coqui_xtts and coqui_xtts.XTTS_AVAILABLE:
                with gr.Tab("Coqui XTTS Fine-Tuning", id=4):
                    with gr.Tabs():
                        with gr.Tab("1 - Data Processing"):
                            ft_out_path = gr.Textbox(label="Output Path", value=os.path.join(os.getcwd(), "xtts_ft_training"), info="Path to save the processed dataset and model checkpoints.")
                            ft_upload_files = gr.File(file_count="multiple", label="Upload Audio Files for Training (.wav, .mp3, .flac)")
                            ft_language = gr.Dropdown(label="Dataset Language", value="en", choices=coqui_xtts.SUPPORTED_LANGUAGES)
                            ft_preprocess_btn = gr.Button("Step 1: Create Dataset", variant="primary")
                            ft_preprocess_status = gr.Label(label="Progress")
                            ft_train_csv = gr.Textbox(label="Train CSV (auto-filled)", interactive=False)
                            ft_eval_csv = gr.Textbox(label="Eval CSV (auto-filled)", interactive=False)

                        with gr.Tab("2 - Fine-tuning"):
                            gr.Markdown("Ensure the Train and Eval CSV paths are filled from the previous step.")
                            ft_num_epochs = gr.Slider(label="Number of Epochs", minimum=1, maximum=100, step=1, value=10)
                            ft_batch_size = gr.Slider(label="Batch Size", minimum=2, maximum=512, step=1, value=4)
                            ft_grad_acumm = gr.Slider(label="Gradient Accumulation Steps", minimum=1, maximum=128, step=1, value=2)
                            ft_max_audio_length = gr.Slider(label="Max Permitted Audio Length (s)", minimum=2, maximum=20, step=1, value=11)
                            ft_train_btn = gr.Button("Step 2: Run Training", variant="primary")
                            ft_train_status = gr.Label(label="Progress")
                            ft_xtts_config = gr.Textbox(label="Fine-tuned Config Path", interactive=False)
                            ft_xtts_vocab = gr.Textbox(label="Fine-tuned Vocab Path", interactive=False)
                            ft_xtts_checkpoint = gr.Textbox(label="Fine-tuned Checkpoint Path", interactive=False)
                            ft_speaker_reference = gr.Textbox(label="Speaker Reference Audio", interactive=False)
                            with gr.Group(visible=False) as ft_tensorboard_group:
                                ft_tensorboard_btn = gr.Button("Launch TensorBoard")
                                ft_tensorboard_status = gr.Textbox(label="TensorBoard Status", interactive=False)
                                ft_tensorboard_url = gr.Markdown(visible=False)
                            
                        with gr.Tab("3 - Inference"):

                            with gr.Row():
                                ft_autofill_btn = gr.Button("Find latest fine-tuned run", variant="secondary")
                                ft_autofill_status = gr.Markdown(visible=False)
                            gr.Markdown("Load the fine-tuned model using the paths from the previous step.")
                            
                            with gr.Row():
                                with gr.Column():
                                    ft_load_btn = gr.Button("Step 3: Load Fine-tuned Model", variant="primary")
                                    ft_load_status = gr.Label(label="Progress")
                                with gr.Column():
                                    ft_tts_language = gr.Dropdown(label="Inference Language", value="en", choices=coqui_xtts.SUPPORTED_LANGUAGES)
                                    ft_tts_btn = gr.Button("Step 4: Generate Speech", variant="primary")
        
                            ft_tts_status = gr.Label(label="Progress")
                            
                            with gr.Row():
                                ft_tts_output_audio = gr.Audio(label="Generated Audio")
                                ft_reference_audio_display = gr.Audio(label="Reference Audio Used")
                            
                            with gr.Row():
                                ft_speaker_reference_upload = gr.Audio(
                                    label="Speaker reference (upload a short WAV/MP3, optional)",
                                    type="filepath"
                                )
                                ft_speaker_reference = gr.Textbox(
                                    label="Speaker reference path (optional)",
                                    placeholder="Path to .wav of the target voice",
                                    interactive=True
                                )

                            # === Fine-Tuning Inference Multi-mode ===
                            gr.Markdown("### Choose how to import text to synthesize")
                            
                            ft_input_mode = gr.Radio(
                                ["UI Text Entry", "Regular Text File (.txt)", "SRT/VTT Subtitle File"],
                                value="UI Text Entry",
                                label="Input Mode",
                                interactive=True
                            )

                            # UI for each mode
                            ft_ui_text_entry = gr.Textbox(
                                label="Enter text here",
                                placeholder="Type or paste the text to synthesize...",
                                visible=True
                            )
                            ft_txt_file_input = gr.File(
                                label="Upload .txt file",
                                file_types=[".txt"],
                                visible=False
                            )
                            ft_srt_vtt_file_input = gr.File(
                                label="Upload .srt or .vtt file",
                                file_types=[".srt", ".vtt"],
                                visible=False
                            )
                            ft_srt_timing_mode = gr.Radio(
                                ["Strict (Cut audio to fit)", "Flexible (Prevent audio cutoff)"],
                                value="Flexible (Prevent audio cutoff)",
                                label="SRT Timing Mode",
                                visible=False
                            )
                        
                            # Toggle visibility
                            def toggle_input_mode(mode):
                                return (
                                    gr.update(visible=(mode == "UI Text Entry")),
                                    gr.update(visible=(mode == "Regular Text File (.txt)")),
                                    gr.update(visible=(mode == "SRT/VTT Subtitle File")),
                                    gr.update(visible=(mode == "SRT/VTT Subtitle File"))
                                )
                        
                            ft_input_mode.change(
                                toggle_input_mode,
                                inputs=ft_input_mode,
                                outputs=[ft_ui_text_entry, ft_txt_file_input, ft_srt_vtt_file_input, ft_srt_timing_mode]
                            )

                            # Run button & outputs


            if coqui_xtts and coqui_xtts.XTTS_AVAILABLE:
                with gr.Tab("Coqui Voice Library", id=2):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Add New Voice to Library")
                            lib_new_voice_audio = gr.Audio(label="Upload Voice Sample", type="filepath")
                            lib_new_voice_name = gr.Textbox(label="Voice Name", placeholder="Enter a unique name for this voice...")
                            lib_save_btn = gr.Button("Save to Library", variant="primary")
                            lib_save_status = gr.Textbox(label="Status", interactive=False)
                        with gr.Column(scale=2):
                            gr.Markdown("## Current Voices")
                            lib_voice_list = gr.Textbox(label="Voices in Library", value="\n".join(coqui_xtts.get_library_voices()), interactive=False, lines=10)
                            lib_refresh_btn = gr.Button("Refresh Library")

            if higgs and higgs.HIGGS_AVAILABLE:
                with gr.Tab("Higgs TTS", id=3):
                    with gr.Accordion("Higgs Configuration Management", open=False):
                        with gr.Row():
                            higgs_config_name = gr.Textbox(label="Config Name", placeholder="Enter name to save settings...")
                            higgs_save_config_btn = gr.Button("Save Config")
                        with gr.Row():
                            higgs_load_config_dd = gr.Dropdown(label="Load Config", choices=higgs.get_higgs_config_files(), scale=3)
                            higgs_load_config_btn = gr.Button("Load", scale=1)
                            higgs_delete_config_btn = gr.Button("Delete", variant="stop", scale=1)
                        higgs_refresh_configs_btn = gr.Button("Refresh Configs")
                        higgs_config_save_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Tabs():
                        with gr.TabItem("Basic & Long-Form"):
                            with gr.Row():
                                with gr.Column():
                                    higgs_lf_transcript = gr.TextArea(label="Transcript", placeholder="Enter text to synthesize...", lines=10)
                                    with gr.Accordion("Voice Options", open=True):
                                        higgs_lf_voice_choice = gr.Radio(choices=["Smart Voice", "Upload Voice", "Predefined Voice"], value="Smart Voice", label="Voice Selection Method")
                                        with gr.Group(visible=False) as higgs_lf_upload_group:
                                            higgs_lf_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                                        with gr.Group(visible=False) as higgs_lf_predefined_group:
                                            higgs_lf_voice_prompt = gr.Dropdown(choices=higgs.higgs_get_all_available_voices(), value="None (Smart Voice)", label="Predefined Voice Prompts")
                                            higgs_lf_refresh_voices = gr.Button("Refresh Voice List")
                                    with gr.Accordion("Generation Parameters", open=False):
                                        higgs_lf_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="Temperature")
                                        higgs_lf_max_new_tokens = gr.Slider(minimum=128, maximum=2048, value=1024, step=128, label="Max New Tokens")
                                        higgs_lf_seed = gr.Number(label="Seed (0 for random)", value=12345, precision=0)
                                        higgs_lf_scene_description = gr.TextArea(label="Scene Description", placeholder="e.g., in a quiet room", lines=2)
                                        higgs_lf_chunk_size = gr.Slider(minimum=100, maximum=500, value=250, step=10, label="Characters per Chunk (for long-form)")
                                    higgs_lf_generate_btn = gr.Button("Generate Audio", variant="primary")
                                with gr.Column():
                                    higgs_lf_output_audio = gr.Audio(label="Generated Audio", type="filepath")

                        with gr.TabItem("Voice Cloning"):
                            with gr.Row():
                                with gr.Column():
                                    higgs_vc_transcript = gr.TextArea(label="Transcript", placeholder="Enter text to synthesize with your voice...", lines=5)
                                    higgs_vc_uploaded_voice = gr.Audio(label="Upload Your Voice Sample (10-30s)", type="numpy")
                                    with gr.Accordion("Generation Parameters", open=False):
                                        higgs_vc_temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.3, step=0.05, label="Temperature")
                                        higgs_vc_max_new_tokens = gr.Slider(minimum=128, maximum=2048, value=1024, step=128, label="Max New Tokens")
                                        higgs_vc_seed = gr.Number(label="Seed (0 for random)", value=12345, precision=0)
                                    higgs_vc_generate_btn = gr.Button("Clone & Generate", variant="primary")
                                with gr.Column():
                                    higgs_vc_output_audio = gr.Audio(label="Cloned Voice Audio", type="filepath")

                        with gr.TabItem("Multi-Speaker & Subtitles"):
                             with gr.Tabs():
                                with gr.TabItem("Multi-Speaker"):
                                    with gr.Row():
                                        with gr.Column():
                                            higgs_ms_transcript = gr.TextArea(label="Multi-Speaker Transcript", placeholder="Use [SPEAKER0], [SPEAKER1] tags...", lines=8)
                                            higgs_ms_auto_format = gr.Checkbox(label="Auto-format dialogue", value=False)
                                            with gr.Accordion("Voice Configuration", open=True):
                                                higgs_ms_voice_method = gr.Radio(choices=["Smart Voice", "Upload Voices", "Predefined Voices"], value="Smart Voice", label="Voice Method")
                                                with gr.Group(visible=False) as higgs_ms_upload_group:
                                                    higgs_ms_speaker0_audio = gr.Audio(label="SPEAKER0 Voice", type="numpy")
                                                    higgs_ms_speaker1_audio = gr.Audio(label="SPEAKER1 Voice", type="numpy")
                                                    higgs_ms_speaker2_audio = gr.Audio(label="SPEAKER2 Voice", type="numpy")
                                                with gr.Group(visible=False) as higgs_ms_predefined_group:
                                                    higgs_ms_speaker0_voice = gr.Dropdown(choices=higgs.higgs_get_all_available_voices(), label="SPEAKER0 Voice")
                                                    higgs_ms_speaker1_voice = gr.Dropdown(choices=higgs.higgs_get_all_available_voices(), label="SPEAKER1 Voice")
                                                    higgs_ms_speaker2_voice = gr.Dropdown(choices=higgs.higgs_get_all_available_voices(), label="SPEAKER2 Voice")
                                                    higgs_ms_refresh_voices_multi = gr.Button("Refresh Voice List")
                                            higgs_ms_generate_btn = gr.Button("Generate Multi-Speaker Audio", variant="primary")
                                        with gr.Column():
                                            higgs_ms_output_audio = gr.Audio(label="Generated Dialogue", type="filepath")

                                with gr.TabItem("Subtitle Generation (.srt/.vtt)"):
                                    with gr.Row():
                                        with gr.Column():
                                            higgs_sub_file_upload = gr.File(label="Upload .srt or .vtt File", file_types=[".srt", ".vtt"])
                                            with gr.Accordion("Voice Options", open=True):
                                                higgs_sub_voice_choice = gr.Radio(choices=["Smart Voice", "Upload Voice", "Predefined Voice"], value="Smart Voice", label="Voice Selection")
                                                with gr.Group(visible=False) as higgs_sub_upload_group:
                                                    higgs_sub_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                                                with gr.Group(visible=False) as higgs_sub_predefined_group:
                                                    higgs_sub_voice_prompt = gr.Dropdown(choices=higgs.higgs_get_all_available_voices(), label="Predefined Voice")
                                                    higgs_sub_refresh_voices_2 = gr.Button("Refresh Voice List")
                                            
                                            higgs_sub_timing_mode = gr.Radio(
                                                label="Subtitle Timing Mode",
                                                choices=["Strict (Cut audio to fit)", "Flexible (Prevent audio cutoff)"],
                                                value="Flexible (Prevent audio cutoff)",
                                                info="Strict mode places audio exactly at subtitle start times, cutting off overflow. Flexible mode adds silence between clips to match start times, preventing cutoff."
                                            )
                                            higgs_sub_generate_btn = gr.Button("Generate Timed Audio", variant="primary")
                                        with gr.Column():
                                            higgs_sub_output_audio = gr.Audio(label="Generated Timed Audio", type="filepath")

                        with gr.TabItem("Higgs Voice Library"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("### Add New Voice to Library")
                                    higgs_vl_new_voice_audio = gr.Audio(label="Upload Voice Sample", type="numpy")
                                    higgs_vl_new_voice_name = gr.Textbox(label="Voice Name", placeholder="Enter a unique name...")
                                    higgs_vl_save_btn = gr.Button("Save to Library", variant="primary")
                                    higgs_vl_save_status = gr.Textbox(label="Status", interactive=False)
                                with gr.Column():
                                    gr.Markdown("### Manage Existing Voices")
                                    higgs_vl_existing_voices = gr.Dropdown(label="Select Voice to Delete", choices=["None"] + higgs.higgs_get_voice_library_voices())
                                    higgs_vl_delete_btn = gr.Button("Delete Selected", variant="stop")
                                    higgs_vl_delete_status = gr.Textbox(label="Delete Status", interactive=False)
                                    higgs_vl_refresh_btn = gr.Button("Refresh Library")
        
        # --- Event Handling ---
        if coqui_xtts and coqui_xtts.XTTS_AVAILABLE:
            clear_cache_button.click(fn=coqui_xtts.clear_tts_cache, outputs=cache_status)
            
        # Whisper Tab Logic
        def handle_whisper_model_change(model_choice):
            if model_choice == "turbo":
                info_text = "⚠️ **Note:** The `turbo` model does not support translation."
                return gr.update(value="transcribe", interactive=False), gr.update(value=info_text, visible=True)
            return gr.update(interactive=True), gr.update(visible=False)
        whisper_model_size.change(fn=handle_whisper_model_change, inputs=whisper_model_size, outputs=[whisper_task, whisper_info_box])

        def handle_whisper_engine_change(engine):
            is_stable = engine == "Stable-TS"
            return gr.update(visible=is_stable), gr.update(visible=is_stable)
        whisper_engine.change(fn=handle_whisper_engine_change, inputs=whisper_engine, outputs=[stable_ts_options, post_processing_options])

        def handle_post_processing_toggle(enabled): return gr.update(visible=enabled)
        post_processing_enabled.change(fn=handle_post_processing_toggle, inputs=post_processing_enabled, outputs=post_processing_group)

        def handle_output_action_change(action): return gr.update(visible="Pipeline" in action)
        whisper_output_action.change(fn=handle_output_action_change, inputs=whisper_output_action, outputs=whisper_pipeline_group)

        def handle_transcription_and_pipeline(*args):
            (
                audio_file_path, model_size, language, task, output_action, whisper_device_val, whisper_engine_val,
                autorun, tts_config_name, tts_device_val,
                regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
                min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
                temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
                suppress_ts_tokens, time_scale,
                post_processing_enabled, refine_enabled, refine_steps, refine_precision,
                remove_repetitions_enabled, remove_repetitions_max_words,
                remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
                split_by_gap_enabled, split_by_gap_value,
                split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
                split_by_duration_enabled, split_by_duration_max_dur,
                merge_by_gap_enabled, merge_by_gap_min_gap, merge_by_gap_max_words,
                merge_by_punctuation_enabled, merge_by_punctuation_string,
                merge_all_segments_enabled,
                fill_gaps_enabled, fill_gaps_file_input, fill_gaps_min_gap
            ) = args

            text_out, preview_out, files_out, tts_input_file_val = run_whisper_transcription(
                audio_file_path, model_size, language, task, output_action, whisper_device_val, whisper_engine_val,
                regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
                min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
                temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
                suppress_ts_tokens, time_scale,
                post_processing_enabled, refine_enabled, refine_steps, refine_precision,
                remove_repetitions_enabled, remove_repetitions_max_words,
                remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
                split_by_gap_enabled, split_by_gap_value,
                split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
                split_by_duration_enabled, split_by_duration_max_dur,
                merge_by_gap_enabled, merge_by_gap_min_gap, merge_by_gap_max_words,
                merge_by_punctuation_enabled, merge_by_punctuation_string,
                merge_all_segments_enabled,
                fill_gaps_enabled, fill_gaps_file_input, fill_gaps_min_gap
            )
            
            if autorun and tts_input_file_val:
                if not (coqui_xtts and getattr(coqui_xtts, "XTTS_AVAILABLE", False)):
                    # XTTS not available → keep outputs shape but inform the user
                    return (text_out, preview_out, files_out, tts_input_file_val, 
                            None, "⚠️ Auto-run skipped: Coqui XTTS is not available.")
                config_path = os.path.join(coqui_xtts.TTS_CONFIG_LIBRARY_PATH, f"{tts_config_name}.json")
                if not os.path.exists(config_path):
                    return (text_out, preview_out, files_out, tts_input_file_val, 
                            None, f"❌ Auto-run failed: Config '{tts_config_name}' not found.")
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                tts_audio_out, tts_status_out = coqui_xtts.run_tts_generation(
                    input_file=tts_input_file_val, language=config_data["language"], voice_mode=config_data["voice_mode"],
                    clone_source=config_data["clone_source"], library_voice=config_data["library_voice"], clone_speaker_audio=None,
                    stock_voice=config_data["stock_voice"], output_format=config_data["output_format"],
                    input_mode=config_data["input_mode"], srt_timing_mode=config_data["srt_timing_mode"], tts_device=tts_device_val
                )
                return text_out, preview_out, files_out, tts_input_file_val, tts_audio_out, tts_status_out
            return text_out, preview_out, files_out, tts_input_file_val, None, ""

        transcribe_btn.click(
            fn=handle_transcription_and_pipeline,
            inputs=[
                whisper_audio_input, whisper_model_size, whisper_language, whisper_task, whisper_output_action, whisper_device, whisper_engine,
                whisper_autorun_tts, whisper_tts_config, tts_device,
                regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
                min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
                temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
                suppress_ts_tokens, time_scale,
                post_processing_enabled, refine_enabled, refine_steps, refine_precision,
                remove_repetitions_enabled, remove_repetitions_max_words,
                remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
                split_by_gap_enabled, split_by_gap_value,
                split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
                split_by_duration_enabled, split_by_duration_max_dur,
                merge_by_gap_enabled, merge_by_gap_min_gap, merge_by_gap_max_words,
                merge_by_punctuation_enabled, merge_by_punctuation_string,
                merge_all_segments_enabled,
                fill_gaps_enabled, fill_gaps_file_input, fill_gaps_min_gap
            ],
            outputs=[whisper_output_text, whisper_file_preview, whisper_output_files, tts_input_file, tts_output_audio, tts_status_textbox]
        ).then(fn=lambda file: gr.update(selected=1) if file else gr.update(), inputs=tts_input_file, outputs=tabs)
        if coqui_xtts and coqui_xtts.XTTS_AVAILABLE:
            # Coqui TTS Tab Logic
            def update_tts_voice_mode(mode): return { tts_clone_voice_group: gr.update(visible=mode == 'Clone'), tts_stock_voice_group: gr.update(visible=mode == 'Stock') }
            tts_voice_mode.change(fn=update_tts_voice_mode, inputs=tts_voice_mode, outputs=[tts_clone_voice_group, tts_stock_voice_group])
            def update_clone_source(source): return { tts_upload_group: gr.update(visible=source == 'Upload New Sample'), tts_library_group: gr.update(visible=source == 'Use from Library') }
            tts_clone_source.change(fn=update_clone_source, inputs=tts_clone_source, outputs=[tts_upload_group, tts_library_group])
            def handle_input_mode_change(mode): return gr.update(visible=mode == "SRT/VTT Mode")
            tts_input_mode.change(fn=handle_input_mode_change, inputs=tts_input_mode, outputs=tts_srt_group)
            tts_generate_btn.click(
                fn=coqui_xtts.run_tts_generation,
                inputs=[tts_input_file, tts_language, tts_voice_mode, tts_clone_source, tts_library_voice, tts_clone_speaker_audio, tts_stock_voice, tts_output_format, tts_input_mode, tts_srt_timing_mode, tts_device, tts_soft_limit, tts_hard_limit],
                outputs=[tts_output_audio, tts_status_textbox]
            )

            # Coqui Voice Library Logic
            def refresh_coqui_library():
                voices = coqui_xtts.get_library_voices()
                return gr.update(choices=voices), gr.update(value="\n".join(voices))
            lib_save_btn.click(fn=coqui_xtts.save_voice_to_library, inputs=[lib_new_voice_audio, lib_new_voice_name], outputs=[lib_save_status]).then(fn=refresh_coqui_library, outputs=[tts_library_voice, lib_voice_list])
            lib_refresh_btn.click(fn=refresh_coqui_library, outputs=[tts_library_voice, lib_voice_list])
            refresh_library_btn_tts.click(fn=refresh_coqui_library, outputs=[tts_library_voice, lib_voice_list])

            # Coqui Fine-Tuning Logic
            if train_module:
                ft_log_dir_state = gr.State()

                def show_tensorboard_button(log_dir):
                    if log_dir and os.path.exists(log_dir):
                        return gr.update(visible=True)
                    return gr.update(visible=False)

                ft_preprocess_btn.click(
                    fn=train_module.preprocess_dataset,
                    inputs=[ft_upload_files, ft_language, ft_out_path],
                    outputs=[ft_preprocess_status, ft_train_csv, ft_eval_csv]
                )
                ft_train_btn.click(
                    fn=train_module.train_model,
                    inputs=[ft_language, ft_train_csv, ft_eval_csv, ft_num_epochs, ft_batch_size, ft_grad_acumm, ft_out_path, ft_max_audio_length],
                    outputs=[ft_train_status, ft_xtts_config, ft_xtts_vocab, ft_xtts_checkpoint, ft_speaker_reference, ft_log_dir_state]
                ).then(
                    fn=show_tensorboard_button,
                    inputs=[ft_log_dir_state],
                    outputs=[ft_tensorboard_group]
                )
                ft_tensorboard_btn.click(
                    fn=train_module.launch_tensorboard,
                    inputs=[ft_log_dir_state],
                    outputs=[ft_tensorboard_status, ft_tensorboard_url]
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=[ft_tensorboard_status]
                )
                
                ft_autofill_btn.click(
                    fn=train_module.autofill_ft_paths,
                    inputs=[ft_out_path],
                    outputs=[ft_xtts_config, ft_xtts_vocab, ft_xtts_checkpoint, ft_speaker_reference, ft_autofill_status]
                )
                ft_load_btn.click(
                    fn=train_module.load_or_discover_model,
                    inputs=[ft_xtts_checkpoint, ft_xtts_config, ft_xtts_vocab, ft_out_path],
                    outputs=[ft_load_status]
                )
                ft_tts_btn.click(
            _run_ft_inference,
            inputs=[ft_ui_text_entry, ft_txt_file_input, ft_srt_vtt_file_input, ft_input_mode, ft_srt_timing_mode, ft_tts_language, ft_speaker_reference_upload],
            outputs=[ft_tts_output_audio, ft_tts_status]
        )

        # Config Refresh Logic
        def refresh_all_config_lists():
            return gr.update(choices=coqui_xtts.get_tts_config_files()), gr.update(choices=get_whisper_config_files())
        if coqui_xtts and getattr(coqui_xtts, "TTS_AVAILABLE", False):
            tts_refresh_configs_btn.click(
                refresh_all_config_lists,
                None, [tts_load_config_dd, whisper_load_config_dd]
            )
            # upgrade whisper refresh to update both, now that TTS widgets exist
            whisper_refresh_configs_btn_main.click(
                refresh_all_config_lists,
                None, [tts_load_config_dd, whisper_load_config_dd]
            )
        whisper_refresh_configs_btn_main.click(
            lambda: gr.update(choices=get_whisper_config_files()),
            None, whisper_load_config_dd
        )

        # Higgs Event Handlers
        if higgs and higgs.HIGGS_AVAILABLE:
            higgs_save_config_btn.click(fn=higgs.save_higgs_config, inputs=[higgs_config_name, higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, higgs_lf_scene_description, higgs_lf_chunk_size, higgs_ms_auto_format], outputs=higgs_config_save_status).then(lambda: gr.update(choices=higgs.get_higgs_config_files()), None, higgs_load_config_dd)
            higgs_load_config_btn.click(fn=higgs.load_higgs_config, inputs=higgs_load_config_dd, outputs=[higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, higgs_lf_scene_description, higgs_lf_chunk_size, higgs_ms_auto_format])
            higgs_delete_config_btn.click(fn=higgs.delete_higgs_config, inputs=higgs_load_config_dd, outputs=higgs_config_save_status).then(lambda: gr.update(choices=higgs.get_higgs_config_files()), None, higgs_load_config_dd)
            higgs_refresh_configs_btn.click(lambda: gr.update(choices=higgs.get_higgs_config_files()), None, higgs_load_config_dd)
            higgs_lf_voice_choice.change(lambda choice: {higgs_lf_upload_group: gr.update(visible=choice == "Upload Voice"), higgs_lf_predefined_group: gr.update(visible=choice == "Predefined Voice")}, higgs_lf_voice_choice, [higgs_lf_upload_group, higgs_lf_predefined_group])
            higgs_lf_generate_btn.click(fn=higgs.higgs_run_longform, inputs=[higgs_lf_transcript, higgs_lf_voice_choice, higgs_lf_uploaded_voice, higgs_lf_voice_prompt, higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, higgs_lf_scene_description, higgs_lf_chunk_size, whisper_device], outputs=higgs_lf_output_audio)
            higgs_lf_refresh_voices.click(lambda: gr.update(choices=higgs.higgs_get_all_available_voices()), None, higgs_lf_voice_prompt)
            higgs_vc_generate_btn.click(fn=higgs.higgs_run_voice_clone, inputs=[higgs_vc_transcript, higgs_vc_uploaded_voice, higgs_vc_temperature, higgs_vc_max_new_tokens, higgs_vc_seed, whisper_device], outputs=higgs_vc_output_audio)
            higgs_ms_voice_method.change(lambda choice: {higgs_ms_upload_group: gr.update(visible=choice == "Upload Voices"), higgs_ms_predefined_group: gr.update(visible=choice == "Predefined Voices")}, higgs_ms_voice_method, [higgs_ms_upload_group, higgs_ms_predefined_group])
            
            # --- FIX: Flatten the inputs list for the click event ---
            higgs_ms_generate_btn.click(
                fn=higgs.higgs_run_multi_speaker, 
                inputs=[
                    higgs_ms_transcript, higgs_ms_voice_method, 
                    higgs_ms_speaker0_audio, higgs_ms_speaker1_audio, higgs_ms_speaker2_audio,
                    higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice,
                    higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, 
                    higgs_lf_scene_description, higgs_ms_auto_format, whisper_device
                ], 
                outputs=higgs_ms_output_audio
            )
            
            def refresh_multi_voice():
                choices = higgs.higgs_get_all_available_voices()
                return gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)
            higgs_ms_refresh_voices_multi.click(refresh_multi_voice, None, [higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice])
            higgs_sub_voice_choice.change(lambda choice: {higgs_sub_upload_group: gr.update(visible=choice == "Upload Voice"), higgs_sub_predefined_group: gr.update(visible=choice == "Predefined Voice")}, higgs_sub_voice_choice, [higgs_sub_upload_group, higgs_sub_predefined_group])
            higgs_sub_generate_btn.click(fn=higgs.higgs_run_subtitle_generation, inputs=[higgs_sub_file_upload, higgs_sub_voice_choice, higgs_sub_uploaded_voice, higgs_sub_voice_prompt, higgs_lf_temperature, higgs_lf_seed, higgs_sub_timing_mode, whisper_device], outputs=higgs_sub_output_audio)
            higgs_sub_refresh_voices_2.click(lambda: gr.update(choices=higgs.higgs_get_all_available_voices()), None, higgs_sub_voice_prompt)
            def refresh_higgs_library_and_prompts():
                lib_voices = ["None"] + higgs.higgs_get_voice_library_voices()
                prompt_voices = higgs.higgs_get_all_available_voices()
                updates = [gr.update(choices=lib_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices)]
                return updates
            higgs_vl_save_btn.click(fn=higgs.higgs_save_voice_to_library, inputs=[higgs_vl_new_voice_audio, higgs_vl_new_voice_name, whisper_device], outputs=higgs_vl_save_status).then(refresh_higgs_library_and_prompts, None, [higgs_vl_existing_voices, higgs_lf_voice_prompt, higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice, higgs_sub_voice_prompt])
            higgs_vl_delete_btn.click(fn=higgs.higgs_delete_voice_from_library, inputs=higgs_vl_existing_voices, outputs=higgs_vl_delete_status).then(refresh_higgs_library_and_prompts, None, [higgs_vl_existing_voices, higgs_lf_voice_prompt, higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice, higgs_sub_voice_prompt])
            higgs_vl_refresh_btn.click(refresh_higgs_library_and_prompts, None, [higgs_vl_existing_voices, higgs_lf_voice_prompt, higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice, higgs_sub_voice_prompt])

    return demo

if __name__ == "__main__":
    if not check_ffmpeg():
        logger.critical("\nHalting execution because FFmpeg is not found.")
        sys.exit(1)
    
    # Try to import Higgs Audio at startup
    logger.info("\n" + "="*60)
    logger.info("🔧 CHECKING HIGGS AUDIO AVAILABILITY")
    logger.info("="*60)
    logger.info("="*60)
    if higgs:
        higgs.try_import_higgs()
        if higgs.HIGGS_AVAILABLE:
            logger.success("🎉 Higgs Audio is available! The Higgs TTS tab will be enabled.")
        else:
         logger.critical("⚠️ Higgs Audio is not available. The Higgs TTS tab will be disabled.")
    else:
        logger.info("ℹ️ Higgs module not found. The Higgs TTS tab will be disabled.")
    
    if coqui_xtts:
        coqui_xtts.check_xtts_availability()
        if coqui_xtts.XTTS_AVAILABLE:
            logger.success("🎉 Coqui XTTS is available! The Coqui XTTS tab will be enabled.")
        else:
            logger.warning("⚠️ Coqui XTTS is not available. The Coqui XTTS tab will be disabled.")

    
    app = create_gradio_ui()
    
    logger.success("\n✅ Gradio UI created. Launching Web UI...")
    logger.info("➡️ Access the UI by opening the 'Running on local URL' link below in your browser.")
    
    app.launch()