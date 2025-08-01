import gradio as gr
import srt
import torch
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
from TTS.utils.manage import ModelManager

# ✨ Import the necessary config classes for PyTorch's safelist
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# NEW: Import stable-ts and its audio module
import stable_whisper
import stable_whisper.audio

# ========================================================================================
# --- Global Model and Configuration ---
# ========================================================================================

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_NAME_FOR_FILE = "Coqui_XTTSv2"
SAMPLE_RATE = 24000
VOICE_LIBRARY_PATH = "voice_library"
TTS_CONFIG_LIBRARY_PATH = "tts_configs"
WHISPER_CONFIG_LIBRARY_PATH = "whisper_configs"

# Global variables to hold the models
tts_model = None
whisper_model = None
stable_whisper_model = None # NEW: Global variable for stable-ts model
current_tts_device = None
current_whisper_device = None
current_whisper_model_size = None
current_whisper_engine = None # NEW: Keep track of the current engine

# --- Device Auto-Detection ---
def get_available_devices():
    """Checks for CUDA availability and returns a list of devices."""
    if torch.cuda.is_available():
        return ["cuda", "cpu"]
    return ["cpu"]

AVAILABLE_DEVICES = get_available_devices()

STOCK_VOICES = {
    'Clarabelle': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/female.wav",
    'Jordan': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/male.wav",
    'Hina': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/hina.wav",
    'William': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/william.wav",
    'Grace': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/grace.wav"
}

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
]

# ========================================================================================
# --- System Prerequisite Check ---
# ========================================================================================
def check_ffmpeg():
    """Checks if FFmpeg is installed and in the system's PATH."""
    if shutil.which("ffmpeg"):
        print("✅ FFmpeg found.")
        return True
    else:
        print("❌ FFmpeg not found.")
        print("This application requires FFmpeg for audio processing.")
        print("Please install it and ensure it's in your system's PATH.")
        print("Installation instructions:")
        print("  - Windows: Download from https://ffmpeg.org/download.html (and add to PATH)")
        print("  - MacOS (via Homebrew): brew install ffmpeg")
        print("  - Linux (Debian/Ubuntu): sudo apt update && sudo apt install ffmpeg")
        return False

# ========================================================================================
# --- Library and Config Functions ---
# ========================================================================================
os.makedirs(VOICE_LIBRARY_PATH, exist_ok=True)
os.makedirs(TTS_CONFIG_LIBRARY_PATH, exist_ok=True)
os.makedirs(WHISPER_CONFIG_LIBRARY_PATH, exist_ok=True)

def clear_tts_cache():
    """Clears the Coqui TTS model cache to force a re-download."""
    global tts_model
    try:
        # This is a robust way to find the cache path used by the TTS library
        # It avoids hardcoding paths and should work across different OSes
        manager = ModelManager()
        # This will trigger the download of a small file to find the root cache path
        manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
        
        # The root path is usually ~/.local/share/tts on Linux/Mac and C:\Users\user\AppData\Local\tts on Windows
        cache_path = os.path.join(os.path.expanduser("~"), ".local", "share", "tts")
        if sys.platform == "win32":
            cache_path = os.path.join(os.getenv("LOCALAPPDATA"), "tts")

        if os.path.exists(cache_path):
            print(f"🚮 Clearing TTS model cache at: {cache_path}")
            shutil.rmtree(cache_path)
            tts_model = None # Unload the model from memory to force a reload
            return f"✅ TTS model cache cleared successfully. The model will be re-downloaded on next use."
        else:
            return "ℹ️ TTS model cache directory not found (it may have already been cleared)."
    except Exception as e:
        # Fallback error message if the above fails
        traceback.print_exc()
        return f"❌ Error clearing TTS cache: {e}. Please check the console for details."

def get_library_voices():
    if not os.path.exists(VOICE_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(VOICE_LIBRARY_PATH) if f.endswith(('.wav', '.mp3'))]

def get_tts_config_files():
    if not os.path.exists(TTS_CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(TTS_CONFIG_LIBRARY_PATH) if f.endswith('.json')]

def get_whisper_config_files():
    if not os.path.exists(WHISPER_CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(WHISPER_CONFIG_LIBRARY_PATH) if f.endswith('.json')]

def save_voice_to_library(audio_filepath, voice_name):
    """Saves an uploaded audio file to the voice library."""
    if audio_filepath is None:
        return "❌ Error: Please upload an audio sample first."
    if not voice_name or not voice_name.strip():
        return "❌ Error: Please enter a name for the voice."

    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", voice_name).strip().replace(" ", "_")
    if not sanitized_name:
        return "❌ Error: The provided voice name is invalid after sanitization."

    destination_path = os.path.join(VOICE_LIBRARY_PATH, f"{sanitized_name}.wav")

    if os.path.exists(destination_path):
        return f"❌ Error: A voice with the name '{sanitized_name}' already exists."

    try:
        shutil.copyfile(audio_filepath, destination_path)
        return f"✅ Voice '{sanitized_name}' saved successfully to the library."
    except Exception as e:
        return f"❌ Error saving voice: {e}"

def save_tts_config(config_name, language, voice_mode, clone_source, library_voice, stock_voice, output_format, input_mode, srt_timing_mode):
    if not config_name or not config_name.strip():
        return "❌ Error: Please enter a name for the configuration."
    
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", config_name).strip().replace(" ", "_")
    if not sanitized_name:
        return "❌ Error: Invalid config name."

    config_path = os.path.join(TTS_CONFIG_LIBRARY_PATH, f"{sanitized_name}.json")
    
    config_data = {
        "language": language, "voice_mode": voice_mode, "clone_source": clone_source,
        "library_voice": library_voice, "stock_voice": stock_voice, "output_format": output_format,
        "input_mode": input_mode, "srt_timing_mode": srt_timing_mode
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=4)
        return f"✅ Config '{sanitized_name}' saved successfully."
    except Exception as e:
        return f"❌ Error saving config: {e}"

def load_tts_config(config_name):
    if not config_name: return [gr.update()]*8
    config_path = os.path.join(TTS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return [gr.update()]*8
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
        return [
            gr.update(value=config_data.get("language")), gr.update(value=config_data.get("voice_mode")),
            gr.update(value=config_data.get("clone_source")), gr.update(value=config_data.get("library_voice")),
            gr.update(value=config_data.get("stock_voice")), gr.update(value=config_data.get("output_format")),
            gr.update(value=config_data.get("input_mode")), gr.update(value=config_data.get("srt_timing_mode"))
        ]
    except Exception as e:
        print(f"Error loading TTS config {config_name}: {e}")
        return [gr.update()]*8

def delete_tts_config(config_name):
    if not config_name: return "ℹ️ No config selected to delete."
    config_path = os.path.join(TTS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return f"❌ Error: Config '{config_name}' not found."
    try:
        os.remove(config_path)
        return f"✅ Config '{config_name}' deleted successfully."
    except Exception as e:
        return f"❌ Error deleting config: {e}"

def save_whisper_config(
    config_name, model_size, language, task, output_action, whisper_engine, autorun, tts_config,
    regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
    min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
    temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
    suppress_ts_tokens, time_scale,
    post_processing_enabled, refine_enabled, refine_steps, refine_precision,
    remove_repetitions_enabled, remove_repetitions_max_words,
    remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
    split_by_gap_enabled, split_by_gap_value,
    split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
    split_by_duration_enabled, split_by_duration_max_dur
):
    if not config_name or not config_name.strip():
        return "❌ Error: Please enter a name for the configuration."
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", config_name).strip().replace(" ", "_")
    if not sanitized_name: return "❌ Error: Invalid config name."

    config_path = os.path.join(WHISPER_CONFIG_LIBRARY_PATH, f"{sanitized_name}.json")
    config_data = {
        "model_size": model_size, "language": language, "task": task,
        "output_action": output_action, "autorun": autorun, "tts_config": tts_config,
        "whisper_engine": whisper_engine,
        "regroup_enabled": regroup_enabled,
        "regroup_string": regroup_string,
        "suppress_silence": suppress_silence,
        "vad_enabled": vad_enabled,
        "vad_threshold": vad_threshold,
        "min_word_dur": min_word_dur,
        "no_speech_threshold": no_speech_threshold,
        "logprob_threshold": logprob_threshold,
        "compression_ratio_threshold": compression_ratio_threshold,
        "temperature": temperature,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": initial_prompt,
        "demucs_enabled": demucs_enabled,
        "only_voice_freq": only_voice_freq,
        "suppress_ts_tokens": suppress_ts_tokens,
        "time_scale": time_scale,
        "post_processing_enabled": post_processing_enabled,
        "refine_enabled": refine_enabled,
        "refine_steps": refine_steps,
        "refine_precision": refine_precision,
        "remove_repetitions_enabled": remove_repetitions_enabled,
        "remove_repetitions_max_words": remove_repetitions_max_words,
        "remove_words_str_enabled": remove_words_str_enabled,
        "words_to_remove": words_to_remove,
        "find_replace_enabled": find_replace_enabled,
        "find_word": find_word,
        "replace_word": replace_word,
        "split_by_gap_enabled": split_by_gap_enabled,
        "split_by_gap_value": split_by_gap_value,
        "split_by_punctuation_enabled": split_by_punctuation_enabled,
        "split_by_length_enabled": split_by_length_enabled,
        "split_by_length_max_chars": split_by_length_max_chars,
        "split_by_length_max_words": split_by_length_max_words,
        "split_by_duration_enabled": split_by_duration_enabled,
        "split_by_duration_max_dur": split_by_duration_max_dur
    }
    try:
        with open(config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=4)
        return f"✅ Config '{sanitized_name}' saved successfully."
    except Exception as e:
        return f"❌ Error saving config: {e}"

def load_whisper_config(config_name):
    if not config_name: return [gr.update()]*39
    config_path = os.path.join(WHISPER_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return [gr.update()]*39
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
        
        updates = [
            gr.update(value=config_data.get("model_size")),
            gr.update(value=config_data.get("language")),
            gr.update(value=config_data.get("task")),
            gr.update(value=config_data.get("output_action")),
            gr.update(value=config_data.get("whisper_engine")),
            gr.update(value=config_data.get("autorun")),
            gr.update(value=config_data.get("tts_config")),
            gr.update(value=config_data.get("regroup_enabled")),
            gr.update(value=config_data.get("regroup_string")),
            gr.update(value=config_data.get("suppress_silence")),
            gr.update(value=config_data.get("vad_enabled")),
            gr.update(value=config_data.get("vad_threshold")),
            gr.update(value=config_data.get("min_word_dur")),
            gr.update(value=config_data.get("no_speech_threshold")),
            gr.update(value=config_data.get("logprob_threshold")),
            gr.update(value=config_data.get("compression_ratio_threshold")),
            gr.update(value=config_data.get("temperature")),
            gr.update(value=config_data.get("condition_on_previous_text")),
            gr.update(value=config_data.get("initial_prompt")),
            gr.update(value=config_data.get("demucs_enabled")),
            gr.update(value=config_data.get("only_voice_freq")),
            gr.update(value=config_data.get("suppress_ts_tokens")),
            gr.update(value=config_data.get("time_scale")),
            gr.update(value=config_data.get("post_processing_enabled")),
            gr.update(value=config_data.get("refine_enabled")),
            gr.update(value=config_data.get("refine_steps")),
            gr.update(value=config_data.get("refine_precision")),
            gr.update(value=config_data.get("remove_repetitions_enabled")),
            gr.update(value=config_data.get("remove_repetitions_max_words")),
            gr.update(value=config_data.get("remove_words_str_enabled")),
            gr.update(value=config_data.get("words_to_remove")),
            gr.update(value=config_data.get("find_replace_enabled")),
            gr.update(value=config_data.get("find_word")),
            gr.update(value=config_data.get("replace_word")),
            gr.update(value=config_data.get("split_by_gap_enabled")),
            gr.update(value=config_data.get("split_by_gap_value")),
            gr.update(value=config_data.get("split_by_punctuation_enabled")),
            gr.update(value=config_data.get("split_by_length_enabled")),
            gr.update(value=config_data.get("split_by_length_max_chars")),
            gr.update(value=config_data.get("split_by_length_max_words")),
            gr.update(value=config_data.get("split_by_duration_enabled")),
            gr.update(value=config_data.get("split_by_duration_max_dur"))
        ]
        
        stable_ts_options_visible = config_data.get("whisper_engine") == "Stable-TS"
        updates.append(gr.update(visible=stable_ts_options_visible))

        post_processing_accordion_visible = config_data.get("post_processing_enabled", False)
        updates.append(gr.update(visible=post_processing_accordion_visible))
        
        return updates
    except Exception as e:
        print(f"Error loading Whisper config {config_name}: {e}")
        return [gr.update()]*39

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

def load_tts_model(device):
    global tts_model, current_tts_device
    if tts_model is None or current_tts_device != device:
        print(f"⏳ Loading Coqui TTS model to device: {device}...")
        try:
            tts_model = TTS(MODEL_NAME, gpu=(device == 'cuda'))
            current_tts_device = device
            print(f"✅ TTS Model loaded successfully on {device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model: {e}")
    return "TTS model is ready."

def load_whisper_model(model_size, device, engine):
    global whisper_model, stable_whisper_model, current_whisper_device, current_whisper_model_size, current_whisper_engine
    
    # Check if the requested model and engine are already loaded
    if current_whisper_model_size == model_size and current_whisper_device == device and current_whisper_engine == engine:
        return "Whisper model is already loaded."

    print(f"⏳ Loading {engine} model '{model_size}' to device: {device}...")
    try:
        if engine == "OpenAI Whisper":
            if stable_whisper_model is not None: stable_whisper_model = None
            whisper_model = whisper.load_model(model_size, device=device)
        elif engine == "Stable-TS":
            if whisper_model is not None: whisper_model = None
            stable_whisper_model = stable_whisper.load_model(model_size, device=device)
        
        current_whisper_device = device
        current_whisper_model_size = model_size
        current_whisper_engine = engine
        print(f"✅ {engine} Model loaded successfully on {device}.")
    except Exception as e:
        raise RuntimeError(f"Failed to load {engine} model: {e}")
    return "Whisper model is ready."

# ========================================================================================
# --- Core Logic Functions ---
# ========================================================================================

def normalize_text(text):
    text = re.sub(r'(\d+)%', lambda m: num2words(int(m.group(1))) + ' percent', text)
    text = re.sub(r'(\d+)x(\d+)', lambda m: f"{num2words(int(m.group(1)))} by {num2words(int(m.group(2)))}", text)
    text = re.sub(r'(\d+)[–—-](\d+)', lambda m: f"{num2words(int(m.group(1)))} to {num2words(int(m.group(2)))}", text)
    text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0))), text)
    return text

def parse_subtitle_file(path):
    with open(path, 'r', encoding='utf-8') as f: return list(srt.parse(f.read()))

def parse_text_file(path):
    with open(path, 'r', encoding='utf-8') as f: lines = [line.strip() for line in f if line.strip()]
    return [srt.Subtitle(index=i, start=timedelta(0), end=timedelta(0), content=line) for i, line in enumerate(lines, 1)]

def create_voiceover(segments, output_path, tts_instance, speaker_wav, language, sample_rate, timed_generation=True, strict_timing=False, progress=gr.Progress()):
    all_audio_chunks = []
    if timed_generation and strict_timing:
        total_duration_seconds = segments[-1].end.total_seconds() if segments else 0
        total_duration_samples = int(total_duration_seconds * sample_rate)
        final_audio = np.zeros(total_duration_samples, dtype=np.float32)
    elif timed_generation and not strict_timing:
        current_time_seconds = 0.0

    for i, sub in enumerate(segments):
        progress((i + 1) / len(segments), desc=f"TTS: Segment {i+1}/{len(segments)}")
        raw_text = sub.content.strip().replace('\n', ' ')
        if not raw_text: continue
        text_to_speak = normalize_text(raw_text)
        try:
            audio_chunk = np.array(tts_instance.tts(text=text_to_speak, speaker_wav=speaker_wav, language=language, split_sentences=False), dtype=np.float32)
            if timed_generation:
                if strict_timing:
                    start_time_sec = sub.start.total_seconds()
                    subtitle_duration_sec = sub.end.total_seconds() - start_time_sec
                    generated_duration_sec = len(audio_chunk) / sample_rate
                    if generated_duration_sec > subtitle_duration_sec: warnings.warn(f"\nLine {sub.index}: Speech ({generated_duration_sec:.2f}s) is LONGER than subtitle duration ({subtitle_duration_sec:.2f}s) and will be CUT OFF.")
                    start_sample = int(start_time_sec * sample_rate)
                    end_sample = start_sample + len(audio_chunk)
                    if end_sample > len(final_audio):
                        audio_chunk = audio_chunk[:len(final_audio) - start_sample]
                        end_sample = len(final_audio)
                    final_audio[start_sample:end_sample] = audio_chunk
                else:
                    chunk_duration_seconds = len(audio_chunk) / sample_rate
                    target_start_time = sub.start.total_seconds()
                    silence_duration = target_start_time - current_time_seconds
                    if silence_duration > 0: all_audio_chunks.append(np.zeros(int(silence_duration * sample_rate), dtype=np.float32))
                    subtitle_duration = sub.end.total_seconds() - sub.start.total_seconds()
                    if chunk_duration_seconds > subtitle_duration: warnings.warn(f"\nLine {sub.index}: Speech ({chunk_duration_seconds:.2f}s) is LONGER than subtitle duration ({subtitle_duration:.2f}s). It may overlap.")
                    all_audio_chunks.append(audio_chunk)
                    current_time_seconds = target_start_time + chunk_duration_seconds
            else:
                all_audio_chunks.append(audio_chunk)
                all_audio_chunks.append(np.zeros(int(0.5 * sample_rate), dtype=np.float32))
        except Exception as e:
            raise e
            
    if not timed_generation or (timed_generation and not strict_timing):
        final_audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)
    
    sf.write(output_path, final_audio, sample_rate)
    return output_path
    
def safe_load_audio(audio_file_path):
    """Loads an audio file into a NumPy array to prevent file path issues."""
    try:
        audio = stable_whisper.audio.load_audio(audio_file_path)
        return audio
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")

# ========================================================================================
# --- Gradio Processing Functions ---
# ========================================================================================

def run_tts_generation(
    input_file, language, voice_mode, clone_source, library_voice, clone_speaker_audio, stock_voice,
    output_format, input_mode, srt_timing_mode, tts_device, progress=gr.Progress(track_tqdm=True)
):
    if input_file is None: return None, "❌ Error: Please upload an input file."
    if voice_mode == 'Clone':
        if clone_source == 'Upload New Sample' and clone_speaker_audio is None: return None, "❌ Error: Please upload a new speaker audio sample."
        if clone_source == 'Use from Library' and not library_voice: return None, "❌ Error: Please select a voice from the library."
    
    try:
        progress(0, desc="Loading TTS Model...")
        load_tts_model(tts_device)
        progress(0.1, desc="Starting TTS Generation...")
        
        speaker_wav_for_tts = None
        if voice_mode == 'Clone':
            if clone_source == 'Upload New Sample': speaker_wav_for_tts = clone_speaker_audio
            else: speaker_wav_for_tts = os.path.join(VOICE_LIBRARY_PATH, f"{library_voice}.wav")
        elif voice_mode == 'Stock':
            voice_url = STOCK_VOICES[stock_voice]
            speaker_wav_for_tts = pooch.retrieve(voice_url, known_hash=None, progressbar=True)
        
        if not os.path.exists(speaker_wav_for_tts): raise FileNotFoundError(f"Speaker reference file not found: {speaker_wav_for_tts}")

        input_filepath = input_file.name if hasattr(input_file, 'name') else input_file
        is_timed = (input_mode == "SRT/VTT Mode")
        segments = parse_subtitle_file(input_filepath) if is_timed else parse_text_file(input_filepath)

        if not segments: return None, "🤷 No processable content found."

        output_dir = "gradio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_filepath))[0]
        clone_or_stock_name = library_voice if (voice_mode == 'Clone' and clone_source == 'Use from Library') else ("clone" if voice_mode == 'Clone' else stock_voice)
        output_filename = f"{base_name}_{clone_or_stock_name}_{language}.{output_format}"
        final_output_path = os.path.join(output_dir, output_filename)
        
        output_audio = create_voiceover(
            segments=segments, output_path=final_output_path, tts_instance=tts_model,
            speaker_wav=speaker_wav_for_tts, language=language, sample_rate=SAMPLE_RATE,
            timed_generation=is_timed, strict_timing=(srt_timing_mode == "Strict (Cut audio to fit)"),
            progress=progress
        )
        
        return output_audio, f"✅ Success! Audio saved to {final_output_path}"

    except Exception as e:
        print("\n--- DETAILED TTS ERROR ---"); traceback.print_exc(); print("--------------------------\n")
        return None, f"❌ An unexpected error occurred: {e}"

def run_whisper_transcription(
    audio_file_path, model_size, language, task, output_action, whisper_device, whisper_engine,
    # Stable-TS parameters
    regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
    min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
    temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
    suppress_ts_tokens, time_scale,
    # Post-Processing options
    post_processing_enabled, refine_enabled, refine_steps, refine_precision,
    remove_repetitions_enabled, remove_repetitions_max_words,
    remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
    split_by_gap_enabled, split_by_gap_value,
    split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
    split_by_duration_enabled, split_by_duration_max_dur,
    progress=gr.Progress(track_tqdm=True)
):
    if audio_file_path is None: return "❌ Error: Please upload an audio file.", "", [], None
    try:
        progress(0, desc=f"Loading {whisper_engine} Model...")
        load_whisper_model(model_size, whisper_device, whisper_engine)
        lang = language if language and language.strip() else None
        
        progress(0.1, desc="Loading audio file...")
        audio_array = safe_load_audio(audio_file_path)

        progress(0.2, desc=f"Starting {whisper_engine} {task}...")
        
        if whisper_engine == "OpenAI Whisper":
            result = whisper_model.transcribe(audio_array, language=lang, task=task, verbose=True)
            full_text = result['text']
            segments = result['segments']
        elif whisper_engine == "Stable-TS":
            regroup_param = regroup_string if regroup_string.strip() else regroup_enabled
            
            result = stable_whisper_model.transcribe(
                audio_array,
                language=lang,
                task=task,
                verbose=True,
                regroup=regroup_param,
                suppress_silence=suppress_silence,
                vad=vad_enabled,
                vad_threshold=vad_threshold,
                min_word_dur=min_word_dur,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt if initial_prompt.strip() else None,
                demucs=demucs_enabled,
                only_voice_freq=only_voice_freq,
                suppress_ts_tokens=suppress_ts_tokens,
                time_scale=time_scale
            )
            
            if post_processing_enabled:
                progress(0.8, desc="Applying Post-Processing...")
                
                # Apply Segmentation
                if split_by_gap_enabled:
                    progress(0.81, desc="Splitting by gap...")
                    result.split_by_gap(split_by_gap_value)
                if split_by_punctuation_enabled:
                    progress(0.82, desc="Splitting by punctuation...")
                    result.split_by_punctuation()
                if split_by_length_enabled:
                    progress(0.83, desc="Splitting by length...")
                    result.split_by_length(split_by_length_max_chars, split_by_length_max_words)
                if split_by_duration_enabled:
                    progress(0.84, desc="Splitting by duration...")
                    result.split_by_duration(split_by_duration_max_dur)

                # Refine Timestamps
                if refine_enabled:
                    progress(0.85, desc="Refining timestamps...")
                    stable_whisper_model.refine(
                        audio_array,
                        result,
                        steps=refine_steps,
                        precision=refine_precision,
                        verbose=False
                    )
                
                # Remove Repetitions
                if remove_repetitions_enabled:
                    progress(0.90, desc="Removing repetitions...")
                    result.remove_repetition(max_words=remove_repetitions_max_words, verbose=False)
                
                # Remove Specific Words by String
                if remove_words_str_enabled and words_to_remove:
                    progress(0.95, desc="Removing specified words...")
                    words_list = [w.strip() for w in words_to_remove.split(',') if w.strip()]
                    result.remove_words_by_str(words_list, verbose=False)

                # Find and Replace
                if find_replace_enabled and find_word and replace_word is not None:
                    progress(0.96, desc="Applying find and replace...")
                    result.replace(find_word, replace_word, case_sensitive=False)
            
            # Extract text and segments after all post-processing
            full_text = result.text
            segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in result.segments]
            
        if output_action == "Display Only": return full_text, "", [], None

        output_dir, base_name, timestamp = "whisper_outputs", os.path.splitext(os.path.basename(audio_file_path))[0], datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        txt_content = full_text
        srt_segments = [srt.Subtitle(i+1, timedelta(seconds=seg['start']), timedelta(seconds=seg['end']), seg['text'].strip()) for i, seg in enumerate(segments)]
        srt_content = srt.compose(srt_segments)
        vtt_content = "WEBVTT\n\n" + "\n\n".join(f"{timedelta(seconds=seg['start'])} --> {timedelta(seconds=seg['end'])}\n{seg['text'].strip()}" for seg in segments)
        json_content = json.dumps({"text": full_text, "segments": segments}, indent=2, ensure_ascii=False)

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
        print("\n--- DETAILED WHISPER ERROR ---"); traceback.print_exc(); print("------------------------------\n")
        return f"❌ An unexpected error occurred: {e}", "", [], None

# ========================================================================================
# --- Gradio UI ---
# ========================================================================================

def create_gradio_ui():
    with gr.Blocks(title="Coqui XTTS & Whisper Pipeline") as demo:
        gr.HTML("""<div style="text-align: center; max-width: 800px; margin: 0 auto;"><h1 style="color: #4CAF50;">Coqui XTTS & Whisper Pipeline</h1><p style="font-size: 1.1em;">A complete toolkit for audio transcription and voiceover generation.</p></div>""")
        
        with gr.Accordion("⚙️ Global Device & Process Settings", open=False):
            with gr.Row():
                tts_device = gr.Radio(label="TTS Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
                whisper_device = gr.Radio(label="Whisper Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
                clear_cache_button = gr.Button("Clear TTS Model Cache", variant="stop")
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
                        
                        whisper_engine = gr.Radio(
                            label="Transcription Engine",
                            choices=["OpenAI Whisper", "Stable-TS"],
                            value="OpenAI Whisper",
                            info="Stable-TS provides more natural, sentence-based segmentation."
                        )
                        
                        whisper_model_size = gr.Dropdown(label="Whisper Model", choices=["tiny", "base", "small", "medium", "large", "turbo"], value="base")
                        whisper_language = gr.Textbox(label="Language (optional)", info="e.g., 'en', 'es'. Leave blank to auto-detect.")
                        whisper_task = gr.Radio(label="Task", choices=["transcribe", "translate"], value="transcribe")
                        gr.Markdown("## 3. Choose Output Action")
                        whisper_output_action = gr.Radio(label="Action", choices=["Display Only", "Save All Formats (.txt, .srt, .vtt, .json)", "Pipeline .txt to TTS", "Pipeline .srt to TTS"], value="Display Only")
                        
                        # NEW: Stable-TS Advanced Options Accordion
                        with gr.Accordion("Stable-TS Advanced Options", open=False, visible=False) as stable_ts_options:
                            gr.Markdown("### Silence Suppression & VAD")
                            suppress_silence = gr.Checkbox(label="Suppress Silence", value=True, info="Adjusts timestamps based on silence detection.")
                            with gr.Row():
                                vad_enabled = gr.Checkbox(label="Use Silero VAD", value=False, info="Use a more accurate Voice Activity Detection model for silence detection.")
                                vad_threshold = gr.Slider(label="VAD Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.35, info="Lower threshold reduces false positives for silence detection.")
                            with gr.Row():
                                min_word_dur = gr.Slider(label="Min Word Duration (s)", minimum=0.0, maximum=1.0, step=0.05, value=0.1, info="Shortest duration a word can be after adjustments.")
                                nonspeech_error = gr.Slider(label="Non-speech Error", minimum=0.0, maximum=1.0, step=0.05, value=0.3, info="Relative error for non-speech sections.")
                            
                            gr.Markdown("### Segmentation & Grouping")
                            with gr.Row():
                                regroup_enabled = gr.Checkbox(label="Enable Regrouping", value=True, info="Enables the default regrouping algorithm for more natural sentences.")
                                regroup_string = gr.Textbox(label="Custom Regrouping Algorithm", placeholder="e.g., cm_sg=.5_mg=.3+3", info="Enter a custom string to override the default regrouping.")

                            gr.Markdown("### Confidence & Decoding")
                            with gr.Row():
                                no_speech_threshold = gr.Slider(label="No Speech Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.6, info="If no-speech probability is above this value, a segment might be considered silent.")
                                logprob_threshold = gr.Slider(label="Log Probability Threshold", minimum=-10.0, maximum=0.0, step=0.5, value=-1.0, info="If average log probability is below this, transcription might be treated as failed.")
                            with gr.Row():
                                compression_ratio_threshold = gr.Slider(label="Compression Ratio Threshold", minimum=0.0, maximum=10.0, step=0.5, value=2.4, info="If gzip compression ratio is above this, transcription might be treated as failed.")
                                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.0, info="Temperature for sampling. Higher values are more random but can be less accurate.")
                            
                            condition_on_previous_text = gr.Checkbox(label="Condition on Previous Text", value=True, info="Provides previous output as a prompt for the next window.")
                            initial_prompt = gr.Textbox(label="Initial Prompt", placeholder="e.g., 'Hello, my name is John.'", info="Text to provide as a prompt for the first window.")
                            
                            gr.Markdown("### Other Options")
                            with gr.Row():
                                only_voice_freq = gr.Checkbox(label="Only Use Voice Frequencies", value=False, info="Filters audio to use only frequencies where most human speech lies (200-5000Hz).")
                                suppress_ts_tokens = gr.Checkbox(label="Suppress Timestamp Tokens", value=False, info="Reduces hallucinations but can ignore disfluencies and repetitions.")
                                demucs_enabled = gr.Checkbox(label="Isolate Vocals (Demucs)", value=False, info="Pre-processes audio with Demucs to isolate vocals. Requires Demucs installation.")
                            time_scale = gr.Slider(label="Time Scale", minimum=0.5, maximum=2.0, step=0.1, value=1.0, info="Factor for scaling audio duration. >1 slows down audio, increasing effective resolution.")

                        with gr.Accordion("Post-Processing", open=False, visible=False) as post_processing_options:
                            post_processing_enabled = gr.Checkbox(label="Enable Post-Processing", value=False, info="Applies a series of functions to the transcription result to improve quality.")
                            
                            with gr.Group(visible=False) as post_processing_group:
                                with gr.Accordion("Refine Timestamps", open=False) as refine_accordion:
                                    refine_enabled = gr.Checkbox(label="Enable Refinement", value=False, info="Iteratively refines word-level timestamps for increased accuracy.")
                                    with gr.Row():
                                        refine_steps = gr.Radio(label="Steps", choices=["se", "s", "e"], value="se", info="Which timestamps to refine: start/end, start only, or end only.")
                                        refine_precision = gr.Slider(label="Precision (s)", minimum=0.02, maximum=1.0, step=0.01, value=0.1, info="Lower values increase precision but also processing time.")

                                with gr.Accordion("Find and Replace", open=False) as find_replace_accordion:
                                    find_replace_enabled = gr.Checkbox(label="Enable Find and Replace", value=False, info="Finds words and replaces them with a specified string.")
                                    find_word = gr.Textbox(label="Find Word/Phrase", placeholder="e.g., 'uh'", info="The word or phrase to find.")
                                    replace_word = gr.Textbox(label="Replace With", placeholder="e.g., 'a'", info="The replacement string.")

                                with gr.Accordion("Split Segments", open=False) as split_accordion:
                                    gr.Markdown("#### Split by Gap")
                                    split_by_gap_enabled = gr.Checkbox(label="Enable Split by Gap", value=False, info="Splits segments with large gaps between words.")
                                    split_by_gap_value = gr.Slider(label="Max Gap (s)", minimum=0.0, maximum=1.0, step=0.01, value=0.1, info="Maximum seconds of silence allowed within a single segment.")
                                    
                                    gr.Markdown("#### Split by Punctuation")
                                    split_by_punctuation_enabled = gr.Checkbox(label="Enable Split by Punctuation", value=False, info="Splits segments at sentence-ending punctuation.")
                                    
                                    gr.Markdown("#### Split by Length")
                                    split_by_length_enabled = gr.Checkbox(label="Enable Split by Length", value=False, info="Splits long segments into smaller ones.")
                                    with gr.Row():
                                        split_by_length_max_chars = gr.Slider(label="Max Chars", minimum=10, maximum=200, step=10, value=50, info="Maximum characters allowed in each segment.")
                                        split_by_length_max_words = gr.Slider(label="Max Words", minimum=5, maximum=50, step=1, value=15, info="Maximum words allowed in each segment.")
                                    
                                    gr.Markdown("#### Split by Duration")
                                    split_by_duration_enabled = gr.Checkbox(label="Enable Split by Duration", value=False, info="Splits segments longer than a certain duration.")
                                    split_by_duration_max_dur = gr.Slider(label="Max Duration (s)", minimum=1, maximum=30, step=1, value=10, info="Maximum duration of each segment in seconds.")

                                with gr.Accordion("Remove Repetitions", open=False) as remove_repetitions_accordion:
                                    remove_repetitions_enabled = gr.Checkbox(label="Enable Repetition Removal", value=False, info="Removes consecutive repeated words.")
                                    remove_repetitions_max_words = gr.Slider(label="Max Words in Repetition", minimum=1, maximum=10, step=1, value=1, info="Maximum number of consecutive words to consider a repetition.")

                                with gr.Accordion("Remove Specific Words", open=False) as remove_words_accordion:
                                    remove_words_str_enabled = gr.Checkbox(label="Enable Word Removal", value=False, info="Removes specific words from the transcript.")
                                    words_to_remove = gr.Textbox(label="Words to Remove (comma-separated)", placeholder="e.g., uh, um, you know", info="Enter words or phrases to be removed.")
                        
                        with gr.Group(visible=False) as whisper_pipeline_group:
                            whisper_autorun_tts = gr.Checkbox(label="Auto-run TTS after pipeline", value=False)
                            whisper_tts_config = gr.Dropdown(label="TTS Config to use", choices=get_tts_config_files())
                            whisper_refresh_tts_configs_btn = gr.Button("Refresh TTS Configs")

                        transcribe_btn = gr.Button("Transcribe Audio", variant="primary")
                        whisper_terminate_btn = gr.Button("Terminate", variant="stop", visible=False)

                    with gr.Column(scale=2):
                        gr.Markdown("## Transcription Result")
                        whisper_output_text = gr.Textbox(label="Output Text", lines=10, interactive=False)
                        with gr.Accordion("File Content Preview & Downloads", open=False):
                            whisper_file_preview = gr.Code(label="File Preview", lines=10, interactive=False)
                            whisper_output_files = gr.File(label="Generated Files", interactive=False)
                        whisper_info_box = gr.Markdown(visible=False)

            with gr.Tab("TTS Voiceover", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Configuration Management", open=False):
                            tts_config_name = gr.Textbox(label="Config Name", placeholder="Enter a name to save current settings...")
                            tts_save_config_btn = gr.Button("Save Config")
                            with gr.Row():
                                tts_load_config_dd = gr.Dropdown(label="Load Config", choices=get_tts_config_files(), scale=3)
                                tts_load_config_btn = gr.Button("Load", scale=1)
                                tts_delete_config_btn = gr.Button("Delete", variant="stop", scale=1)
                            tts_refresh_configs_btn = gr.Button("Refresh Configs")
                            tts_config_save_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("## 1. Upload Your Content")
                        tts_input_file = gr.File(label="Input File (.txt, .srt, .vtt)", file_types=['.txt', '.srt', '.vtt'])
                        tts_input_mode = gr.Radio(label="Input Mode", choices=["Default", "SRT/VTT Mode"], value="Default")
                        
                        gr.Markdown("## 2. Configure Voice")
                        tts_voice_mode = gr.Radio(label="Voice Mode", choices=['Clone', 'Stock'], value='Stock')
                        with gr.Group(visible=False) as tts_clone_voice_group:
                            tts_clone_source = gr.Radio(label="Clone Source", choices=["Upload New Sample", "Use from Library"], value="Upload New Sample")
                            with gr.Group(visible=True) as tts_upload_group: tts_clone_speaker_audio = gr.Audio(label="Upload Voice Sample (6-30s)", type="filepath")
                            with gr.Group(visible=False) as tts_library_group:
                                tts_library_voice = gr.Dropdown(label="Select Library Voice", choices=get_library_voices())
                                refresh_library_btn_tts = gr.Button("Refresh Library")
                        with gr.Group(visible=True) as tts_stock_voice_group: tts_stock_voice = gr.Dropdown(label="Stock Voice", choices=list(STOCK_VOICES.keys()), value='Clarabelle')
                        
                        gr.Markdown("## 3. Configure Output")
                        tts_language = gr.Dropdown(label="Language", choices=SUPPORTED_LANGUAGES, value="en")
                        tts_output_format = gr.Radio(label="Output Format", choices=['wav', 'mp3'], value='wav')
                        with gr.Group(visible=False) as tts_srt_group:
                            tts_srt_timing_mode = gr.Radio(label="SRT/VTT Timing Mode", choices=["Strict (Cut audio to fit)", "Flexible (Prevent audio cutoff)"], value="Flexible (Prevent audio cutoff)")
                        
                        tts_generate_btn = gr.Button("Generate Voiceover", variant="primary")
                        tts_terminate_btn = gr.Button("Terminate", variant="stop", visible=False)

                    with gr.Column(scale=2):
                        gr.Markdown("## Generated Audio")
                        tts_output_audio = gr.Audio(label="Output", type="filepath", show_download_button=True)
                        tts_status_textbox = gr.Textbox(label="Status", interactive=False)

            with gr.Tab("Voice Library", id=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Add New Voice to Library")
                        lib_new_voice_audio = gr.Audio(label="Upload Voice Sample", type="filepath")
                        lib_new_voice_name = gr.Textbox(label="Voice Name", placeholder="Enter a unique name for this voice...")
                        lib_save_btn = gr.Button("Save to Library", variant="primary")
                        lib_save_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown("## Current Voices")
                        lib_voice_list = gr.Textbox(label="Voices in Library", value="\n".join(get_library_voices()), interactive=False, lines=10)
                        lib_refresh_btn = gr.Button("Refresh Library")
        
        # --- Event Handling ---
        
        clear_cache_button.click(fn=clear_tts_cache, outputs=cache_status)

        def handle_whisper_model_change(model_choice):
            if model_choice == "turbo":
                info_text = "⚠️ **Note:** The `turbo` model does not support translation."
                return gr.update(value="transcribe", interactive=False), gr.update(value=info_text, visible=True)
            return gr.update(interactive=True), gr.update(visible=False)
        whisper_model_size.change(fn=handle_whisper_model_change, inputs=whisper_model_size, outputs=[whisper_task, whisper_info_box])
        
        def handle_whisper_engine_change(engine):
            stable_options_visibility = engine == "Stable-TS"
            post_processing_options_visibility = stable_options_visibility
            return gr.update(visible=stable_options_visibility), gr.update(visible=post_processing_options_visibility)
        whisper_engine.change(fn=handle_whisper_engine_change, inputs=whisper_engine, outputs=[stable_ts_options, post_processing_options])

        def handle_post_processing_toggle(enabled):
            return gr.update(visible=enabled)
        post_processing_enabled.change(fn=handle_post_processing_toggle, inputs=post_processing_enabled, outputs=post_processing_group)

        def handle_output_action_change(action):
            return gr.update(visible="Pipeline" in action)
        whisper_output_action.change(fn=handle_output_action_change, inputs=whisper_output_action, outputs=whisper_pipeline_group)

        def update_tts_voice_mode(mode): return { tts_clone_voice_group: gr.update(visible=mode == 'Clone'), tts_stock_voice_group: gr.update(visible=mode == 'Stock') }
        tts_voice_mode.change(fn=update_tts_voice_mode, inputs=tts_voice_mode, outputs=[tts_clone_voice_group, tts_stock_voice_group])
        
        def update_clone_source(source): return { tts_upload_group: gr.update(visible=source == 'Upload New Sample'), tts_library_group: gr.update(visible=source == 'Use from Library') }
        tts_clone_source.change(fn=update_clone_source, inputs=tts_clone_source, outputs=[tts_upload_group, tts_library_group])
        
        def handle_input_mode_change(mode):
            return gr.update(visible=mode == "SRT/VTT Mode")
        tts_input_mode.change(fn=handle_input_mode_change, inputs=tts_input_mode, outputs=tts_srt_group)

        def refresh_all_voice_lists():
            voices = get_library_voices()
            return gr.update(choices=voices), gr.update(value="\n".join(voices))
        lib_save_btn.click(fn=save_voice_to_library, inputs=[lib_new_voice_audio, lib_new_voice_name], outputs=[lib_save_status]).then(fn=refresh_all_voice_lists, outputs=[tts_library_voice, lib_voice_list])
        lib_refresh_btn.click(fn=refresh_all_voice_lists, outputs=[tts_library_voice, lib_voice_list])
        refresh_library_btn_tts.click(fn=refresh_all_voice_lists, outputs=[tts_library_voice, lib_voice_list])

        def refresh_all_config_lists():
            tts_configs, whisper_configs = get_tts_config_files(), get_whisper_config_files()
            return gr.update(choices=tts_configs), gr.update(choices=tts_configs), gr.update(choices=whisper_configs)
        
        tts_save_config_btn.click(fn=save_tts_config, inputs=[tts_config_name, tts_language, tts_voice_mode, tts_clone_source, tts_library_voice, tts_stock_voice, tts_output_format, tts_input_mode, tts_srt_timing_mode], outputs=tts_config_save_status).then(fn=refresh_all_config_lists, outputs=[tts_load_config_dd, whisper_tts_config, whisper_load_config_dd])
        tts_load_config_btn.click(fn=load_tts_config, inputs=tts_load_config_dd, outputs=[tts_language, tts_voice_mode, tts_clone_source, tts_library_voice, tts_stock_voice, tts_output_format, tts_input_mode, tts_srt_timing_mode])
        tts_delete_config_btn.click(fn=delete_tts_config, inputs=tts_load_config_dd, outputs=tts_config_save_status).then(fn=refresh_all_config_lists, outputs=[tts_load_config_dd, whisper_tts_config, whisper_load_config_dd])
        tts_refresh_configs_btn.click(fn=refresh_all_config_lists, outputs=[tts_load_config_dd, whisper_tts_config, whisper_load_config_dd])
        
        whisper_save_config_btn.click(
            fn=save_whisper_config, 
            inputs=[
                whisper_config_name, whisper_model_size, whisper_language, whisper_task, whisper_output_action, whisper_engine, whisper_autorun_tts, whisper_tts_config,
                regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
                min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
                temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
                suppress_ts_tokens, time_scale,
                post_processing_enabled, refine_enabled, refine_steps, refine_precision,
                remove_repetitions_enabled, remove_repetitions_max_words,
                remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
                split_by_gap_enabled, split_by_gap_value,
                split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
                split_by_duration_enabled, split_by_duration_max_dur
            ], 
            outputs=whisper_config_save_status
        ).then(
            fn=refresh_all_config_lists, 
            outputs=[tts_load_config_dd, whisper_tts_config, whisper_load_config_dd]
        )
        
        whisper_load_config_btn.click(
            fn=load_whisper_config, 
            inputs=whisper_load_config_dd, 
            outputs=[
                whisper_model_size, whisper_language, whisper_task, whisper_output_action, whisper_engine, whisper_autorun_tts, whisper_tts_config,
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
                stable_ts_options, post_processing_options
            ]
        )
        
        whisper_delete_config_btn.click(fn=delete_whisper_config, inputs=whisper_load_config_dd, outputs=whisper_config_save_status).then(fn=refresh_all_config_lists, outputs=[tts_load_config_dd, whisper_tts_config, whisper_load_config_dd])
        whisper_refresh_configs_btn_main.click(fn=refresh_all_config_lists, outputs=[tts_load_config_dd, whisper_tts_config, whisper_load_config_dd])
        whisper_refresh_tts_configs_btn.click(fn=refresh_all_config_lists, outputs=[tts_load_config_dd, whisper_tts_config, whisper_load_config_dd])

        def handle_transcription_and_pipeline(
            audio_file_path, model_size, language, task, output_action, whisper_device_value, whisper_engine_value,
            autorun, tts_config_name, tts_device_value,
            # Stable-TS parameters
            regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
            min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
            temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
            suppress_ts_tokens, time_scale,
            # Post-Processing options
            post_processing_enabled, refine_enabled, refine_steps, refine_precision,
            remove_repetitions_enabled, remove_repetitions_max_words,
            remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
            split_by_gap_enabled, split_by_gap_value,
            split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
            split_by_duration_enabled, split_by_duration_max_dur,
            progress=gr.Progress(track_tqdm=True)
        ):
            text_out, preview_out, files_out, tts_input_file_val = run_whisper_transcription(
                audio_file_path, model_size, language, task, output_action, whisper_device_value, whisper_engine_value,
                # Stable-TS parameters passed here
                regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
                min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
                temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
                suppress_ts_tokens, time_scale,
                # Post-Processing parameters passed here
                post_processing_enabled, refine_enabled, refine_steps, refine_precision,
                remove_repetitions_enabled, remove_repetitions_max_words,
                remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
                split_by_gap_enabled, split_by_gap_value,
                split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
                split_by_duration_enabled, split_by_duration_max_dur,
                progress
            )
            
            if autorun and tts_input_file_val:
                progress(0.9, desc="Auto-running TTS...")
                config_path = os.path.join(TTS_CONFIG_LIBRARY_PATH, f"{tts_config_name}.json")
                if not os.path.exists(config_path):
                    return text_out, preview_out, files_out, tts_input_file_val, None, f"❌ Auto-run failed: Config '{tts_config_name}' not found."
                
                with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
                
                tts_audio_out, tts_status_out = run_tts_generation(
                    input_file=tts_input_file_val, language=config_data["language"], voice_mode=config_data["voice_mode"],
                    clone_source=config_data["clone_source"], library_voice=config_data["library_voice"], clone_speaker_audio=None,
                    stock_voice=config_data["stock_voice"], output_format=config_data["output_format"],
                    input_mode=config_data["input_mode"], srt_timing_mode=config_data["srt_timing_mode"], tts_device=tts_device_value, progress=progress
                )
                return text_out, preview_out, files_out, tts_input_file_val, tts_audio_out, tts_status_out

            return text_out, preview_out, files_out, tts_input_file_val, None, ""

        # --- Button Click Handlers ---
        
        tts_click_event = tts_generate_btn.click(
            fn=run_tts_generation,
            inputs=[
                tts_input_file, tts_language, tts_voice_mode, tts_clone_source, tts_library_voice,
                tts_clone_speaker_audio, tts_stock_voice, tts_output_format, tts_input_mode, tts_srt_timing_mode, tts_device
            ],
            outputs=[tts_output_audio, tts_status_textbox]
        )
        tts_terminate_btn.click(fn=None, cancels=[tts_click_event])

        whisper_click_event = transcribe_btn.click(
            fn=handle_transcription_and_pipeline,
            inputs=[
                whisper_audio_input, whisper_model_size, whisper_language, whisper_task, whisper_output_action, whisper_device, whisper_engine,
                whisper_autorun_tts, whisper_tts_config, tts_device,
                # Stable-TS options
                regroup_enabled, regroup_string, suppress_silence, vad_enabled, vad_threshold,
                min_word_dur, no_speech_threshold, logprob_threshold, compression_ratio_threshold,
                temperature, condition_on_previous_text, initial_prompt, demucs_enabled, only_voice_freq,
                suppress_ts_tokens, time_scale,
                # Post-Processing inputs
                post_processing_enabled, refine_enabled, refine_steps, refine_precision,
                remove_repetitions_enabled, remove_repetitions_max_words,
                remove_words_str_enabled, words_to_remove, find_replace_enabled, find_word, replace_word,
                split_by_gap_enabled, split_by_gap_value,
                split_by_punctuation_enabled, split_by_length_enabled, split_by_length_max_chars, split_by_length_max_words,
                split_by_duration_enabled, split_by_duration_max_dur
            ],
            outputs=[whisper_output_text, whisper_file_preview, whisper_output_files, tts_input_file, tts_output_audio, tts_status_textbox]
        ).then(fn=lambda file: gr.update(selected=1) if file else gr.update(), inputs=tts_input_file, outputs=tabs)
        whisper_terminate_btn.click(fn=None, cancels=[whisper_click_event])

    return demo

if __name__ == "__main__":
    if not check_ffmpeg():
        sys.exit(1)

    app = create_gradio_ui()
    
    print("\n✅ Gradio UI created. Launching Web UI...")
    print("➡️ Access the UI by opening the 'Running on local URL' link below in your browser.")
    
    app.launch(share=True)
