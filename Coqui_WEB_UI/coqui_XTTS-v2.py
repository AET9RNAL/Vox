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

# Coqui Imports
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Stable-TS Imports
import stable_whisper
import stable_whisper.audio

# Higgs Audio Imports
try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    HIGGS_AVAILABLE = True
except ImportError:
    HIGGS_AVAILABLE = False
    print("⚠️ Higgs Audio not found. Please run the setup-run.bat script to install it.")
    print("⚠️ The 'Higgs TTS' tab will be disabled.")

# Faster Whisper for Higgs
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("✅ Using faster-whisper for Higgs transcription")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("⚠️ faster-whisper not available for Higgs - voice samples will use dummy text")


# ========================================================================================
# --- Global Model and Configuration ---
# ========================================================================================

# Coqui Config
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_NAME_FOR_FILE = "Coqui_XTTSv2"
SAMPLE_RATE = 24000
VOICE_LIBRARY_PATH = "voice_library"
TTS_CONFIG_LIBRARY_PATH = "tts_configs"
WHISPER_CONFIG_LIBRARY_PATH = "whisper_configs"

# Higgs Config
HIGGS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
HIGGS_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
HIGGS_CONFIG_LIBRARY_PATH = "higgs_configs"
HIGGS_VOICE_LIBRARY_PATH = "higgs_voice_library"

# Global variables to hold the models
tts_model = None
whisper_model = None
stable_whisper_model = None 
higgs_serve_engine = None # Higgs Model
higgs_whisper_model = None # Separate whisper for Higgs if needed

current_tts_device = None
current_whisper_device = None
current_whisper_model_size = None
current_whisper_engine = None
current_higgs_device = None

# --- Device Auto-Detection ---
def get_available_devices():
    """Checks for CUDA availability and returns a list of devices."""
    if torch.cuda.is_available():
        return ["cuda", "cpu"]
    return ["cpu"]

AVAILABLE_DEVICES = get_available_devices()

# Coqui Stock Voices
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
    if shutil.which("ffmpeg") or pydub_which("ffmpeg"):
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
# --- Library and Config Functions (Shared & Specific) ---
# ========================================================================================
os.makedirs(VOICE_LIBRARY_PATH, exist_ok=True)
os.makedirs(TTS_CONFIG_LIBRARY_PATH, exist_ok=True)
os.makedirs(WHISPER_CONFIG_LIBRARY_PATH, exist_ok=True)
os.makedirs(HIGGS_CONFIG_LIBRARY_PATH, exist_ok=True)
os.makedirs(HIGGS_VOICE_LIBRARY_PATH, exist_ok=True)
os.makedirs("gradio_outputs", exist_ok=True)
os.makedirs("whisper_outputs", exist_ok=True)
os.makedirs("higgs_outputs/basic_generation", exist_ok=True)
os.makedirs("higgs_outputs/voice_cloning", exist_ok=True)
os.makedirs("higgs_outputs/longform_generation", exist_ok=True)
os.makedirs("higgs_outputs/multi_speaker", exist_ok=True)
os.makedirs("higgs_outputs/subtitle_generation", exist_ok=True)


def clear_tts_cache():
    # ... (existing function, no changes needed)
    global tts_model
    try:
        cache_path = os.path.join(os.path.expanduser("~"), ".local", "share", "tts")
        if sys.platform == "win32":
            cache_path = os.path.join(os.getenv("LOCALAPPDATA"), "tts")

        if os.path.exists(cache_path):
            print(f"🚮 Clearing TTS model cache at: {cache_path}")
            shutil.rmtree(cache_path)
            tts_model = None 
            return f"✅ TTS model cache cleared successfully."
        else:
            return "ℹ️ TTS model cache directory not found."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error clearing TTS cache: {e}."

# --- Coqui XTTS Configs ---
def get_library_voices():
    # ... (existing function)
    if not os.path.exists(VOICE_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(VOICE_LIBRARY_PATH) if f.endswith(('.wav', '.mp3'))]

def get_tts_config_files():
    # ... (existing function)
    if not os.path.exists(TTS_CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(TTS_CONFIG_LIBRARY_PATH) if f.endswith('.json')]

# ... (All other existing config functions for Coqui XTTS and Whisper remain unchanged)
def save_tts_config(config_name, language, voice_mode, clone_source, library_voice, stock_voice, output_format, input_mode, srt_timing_mode):
    if not config_name or not config_name.strip():
        return "❌ Error: Please enter a name for the configuration."
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", config_name).strip().replace(" ", "_")
    config_path = os.path.join(TTS_CONFIG_LIBRARY_PATH, f"{sanitized_name}.json")
    config_data = { "language": language, "voice_mode": voice_mode, "clone_source": clone_source, "library_voice": library_voice, "stock_voice": stock_voice, "output_format": output_format, "input_mode": input_mode, "srt_timing_mode": srt_timing_mode }
    with open(config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=4)
    return f"✅ Config '{sanitized_name}' saved."

def load_tts_config(config_name):
    if not config_name: return [gr.update()]*8
    config_path = os.path.join(TTS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return [gr.update()]*8
    with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
    return [ gr.update(value=config_data.get(k)) for k in ["language", "voice_mode", "clone_source", "library_voice", "stock_voice", "output_format", "input_mode", "srt_timing_mode"]]

def delete_tts_config(config_name):
    if not config_name: return "ℹ️ No config selected."
    config_path = os.path.join(TTS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if os.path.exists(config_path):
        os.remove(config_path)
        return f"✅ Config '{config_name}' deleted."
    return f"❌ Error: Config '{config_name}' not found."
    
# --- Whisper Configs ---
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
# ... (save_whisper_config, load_whisper_config, delete_whisper_config) are unchanged
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
    




# --- NEW: Higgs TTS Configs ---
def get_higgs_config_files():
    if not os.path.exists(HIGGS_CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(HIGGS_CONFIG_LIBRARY_PATH) if f.endswith('.json')]

def save_higgs_config(config_name, temperature, max_new_tokens, seed, scene_description, chunk_size, auto_format):
    if not config_name or not config_name.strip():
        return "❌ Error: Please enter a name for the configuration."
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", config_name).strip().replace(" ", "_")
    config_path = os.path.join(HIGGS_CONFIG_LIBRARY_PATH, f"{sanitized_name}.json")
    config_data = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "scene_description": scene_description,
        "chunk_size": chunk_size,
        "auto_format": auto_format,
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    return f"✅ Higgs Config '{sanitized_name}' saved."

def load_higgs_config(config_name):
    if not config_name: return [gr.update()]*6
    config_path = os.path.join(HIGGS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return [gr.update()]*6
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return [
        gr.update(value=config_data.get("temperature")),
        gr.update(value=config_data.get("max_new_tokens")),
        gr.update(value=config_data.get("seed")),
        gr.update(value=config_data.get("scene_description")),
        gr.update(value=config_data.get("chunk_size")),
        gr.update(value=config_data.get("auto_format")),
    ]

def delete_higgs_config(config_name):
    if not config_name: return "ℹ️ No config selected to delete."
    config_path = os.path.join(HIGGS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if os.path.exists(config_path):
        os.remove(config_path)
        return f"✅ Higgs Config '{config_name}' deleted."
    return f"❌ Error: Config '{config_name}' not found."


# ========================================================================================
# --- Model Loading Functions ---
# ========================================================================================

def load_tts_model(device):
    # ... (existing function, no changes needed)
    global tts_model, current_tts_device
    if tts_model is None or current_tts_device != device:
        print(f"⏳ Loading Coqui TTS model to device: {device}...")
        tts_model = TTS(MODEL_NAME, gpu=(device == 'cuda'))
        current_tts_device = device
        print(f"✅ TTS Model loaded successfully on {device}.")
    return "TTS model is ready."

def load_whisper_model(model_size, device, engine):
    # ... (existing function, no changes needed)
    global whisper_model, stable_whisper_model, current_whisper_device, current_whisper_model_size, current_whisper_engine
    if current_whisper_model_size == model_size and current_whisper_device == device and current_whisper_engine == engine:
        return "Whisper model is already loaded."
    print(f"⏳ Loading {engine} model '{model_size}' to device: {device}...")
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
    return "Whisper model is ready."

def load_higgs_model(device):
    """Loads the Higgs Audio model."""
    global higgs_serve_engine, current_higgs_device
    if not HIGGS_AVAILABLE:
        raise gr.Error("Higgs Audio library not installed. Please run setup-run.bat.")
    if higgs_serve_engine is None or current_higgs_device != device:
        print(f"⏳ Loading Higgs Audio model to device: {device}...")
        try:
            higgs_serve_engine = HiggsAudioServeEngine(HIGGS_MODEL_PATH, HIGGS_AUDIO_TOKENIZER_PATH, device=device)
            current_higgs_device = device
            print(f"✅ Higgs Audio Model loaded successfully on {device}.")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Higgs Audio model: {e}")
    return "Higgs Audio model is ready."

def load_higgs_whisper_model(device):
    """Loads the faster-whisper model for Higgs."""
    global higgs_whisper_model
    if not FASTER_WHISPER_AVAILABLE:
        return
    if higgs_whisper_model is None:
        print("⏳ Loading faster-whisper model for Higgs...")
        try:
            # Using a smaller model for faster voice library creation
            higgs_whisper_model = WhisperModel("base", device=device, compute_type="float16" if device == "cuda" else "int8")
            print("✅ faster-whisper model loaded.")
        except Exception as e:
            print(f"❌ Failed to load faster-whisper model: {e}")


# ========================================================================================
# --- Higgs Audio Helper Functions (prefixed with higgs_) ---
# ========================================================================================
def higgs_save_temp_audio_robust(audio_data, sample_rate):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    if isinstance(audio_data, np.ndarray):
        waveform = torch.from_numpy(audio_data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(temp_path, waveform, sample_rate)
    return temp_path

def higgs_process_uploaded_audio(uploaded_audio):
    if uploaded_audio is None: return None, None
    sample_rate, audio_data = uploaded_audio
    if isinstance(audio_data, np.ndarray):
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        if len(audio_data.shape) > 1: # is stereo
             audio_data = np.mean(audio_data, axis=1) # convert to mono
        return audio_data, sample_rate
    return None, None

def higgs_save_temp_audio_fixed(uploaded_voice):
    if uploaded_voice is None: return None
    processed_audio, processed_rate = higgs_process_uploaded_audio(uploaded_voice)
    if processed_audio is not None:
        return higgs_save_temp_audio_robust(processed_audio, processed_rate)
    return None

def higgs_transcribe_audio(audio_path, device):
    """Transcribe audio file to text using faster-whisper."""
    if not FASTER_WHISPER_AVAILABLE:
        return "This is a voice sample for cloning."
    try:
        load_higgs_whisper_model(device)
        if higgs_whisper_model is None:
             return "This is a voice sample for cloning."
        segments, _ = higgs_whisper_model.transcribe(audio_path, language="en")
        transcription = " ".join([segment.text for segment in segments])
        transcription = transcription.strip()
        if not transcription:
            transcription = "This is a voice sample for cloning."
        print(f"🎤 Higgs Transcribed: {transcription[:100]}...")
        return transcription
    except Exception as e:
        print(f"❌ Higgs Transcription failed: {e}")
        return "This is a voice sample for cloning."

def higgs_create_voice_reference_txt(audio_path, device, transcript_sample=None):
    base_path, _ = os.path.splitext(audio_path)
    txt_path = base_path + '.txt'
    if transcript_sample is None:
        transcript_sample = higgs_transcribe_audio(audio_path, device)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcript_sample)
    return txt_path

def higgs_robust_txt_path_creation(audio_path):
    base_path, _ = os.path.splitext(audio_path)
    return base_path + '.txt'

def higgs_robust_file_cleanup(files):
    if not files: return
    if isinstance(files, str): files = [files]
    for f in files:
        if f and isinstance(f, str) and os.path.exists(f):
            try: os.remove(f)
            except Exception: pass

def higgs_get_output_path(category, filename_base, extension=".wav"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename_base}{extension}"
    return os.path.join("higgs_outputs", category, filename)

def higgs_get_voice_library_voices():
    voices = []
    if os.path.exists(HIGGS_VOICE_LIBRARY_PATH):
        for f in os.listdir(HIGGS_VOICE_LIBRARY_PATH):
            if f.endswith('.wav'):
                voices.append(os.path.splitext(f)[0])
    return voices

def higgs_get_all_available_voices():
    library = higgs_get_voice_library_voices()
    combined = ["None (Smart Voice)"]
    if library:
        combined.extend([f"👤 {voice}" for voice in library])
    return combined

def higgs_get_voice_path(voice_selection):
    if not voice_selection or voice_selection == "None (Smart Voice)":
        return None
    if voice_selection.startswith("👤 "):
        voice_name = voice_selection[2:]
        return os.path.join(HIGGS_VOICE_LIBRARY_PATH, f"{voice_name}.wav")
    return None

def higgs_smart_chunk_text(text, max_chunk_size=200):
    paragraphs = text.split('\n\n')
    chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph: continue
        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
            continue
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        if current_chunk: chunks.append(current_chunk)
    return chunks

def higgs_parse_multi_speaker_text(text):
    speaker_pattern = r'\[SPEAKER(\d+)\]\s*([^[]*?)(?=\[SPEAKER\d+\]|$)'
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    speakers = {}
    for speaker_id, content in matches:
        speaker_key = f"SPEAKER{speaker_id}"
        if speaker_key not in speakers:
            speakers[speaker_key] = []
        speakers[speaker_key].append(content.strip())
    return speakers

def higgs_auto_format_multi_speaker(text):
    if '[SPEAKER' in text: return text
    lines = text.split('\n')
    formatted_lines = []
    current_speaker = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('"') or line.startswith("'") or ':' in line:
            if formatted_lines: current_speaker = 1 - current_speaker
        formatted_lines.append(f"[SPEAKER{current_speaker}] {line}")
    return '\n'.join(formatted_lines)


# ========================================================================================
# --- Core Logic Functions (Coqui XTTS) ---
# ========================================================================================
# ... (All existing Coqui XTTS core logic functions remain unchanged)
# normalize_text, parse_subtitle_file, parse_text_file, create_voiceover, etc.
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

# NEW: Custom function for find and replace
def find_and_replace(result, find_word, replace_word):
    if not find_word or replace_word is None:
        return result
    for segment in result.segments:
        for word in segment.words:
            if word.word.strip().lower() == find_word.strip().lower():
                word.word = f" {replace_word} "
    return result
# ========================================================================================
# --- Gradio Processing Functions (Coqui XTTS) ---
# ========================================================================================
# ... (All existing Coqui XTTS Gradio processing functions remain unchanged)
# run_tts_generation, run_whisper_transcription, etc.
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
                    result = find_and_replace(result, find_word, replace_word)
            
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
# --- Gradio Processing Functions (Higgs TTS) ---
# ========================================================================================
def higgs_run_basic_generation(transcript, voice_prompt, temperature, max_new_tokens, seed, scene_description, device):
    load_higgs_model(device)
    if seed > 0: torch.manual_seed(seed)
    
    system_content = "Generate audio following instruction."
    if scene_description: system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
    
    ref_audio_path = higgs_get_voice_path(voice_prompt)
    if ref_audio_path and os.path.exists(ref_audio_path):
        txt_path = higgs_robust_txt_path_creation(ref_audio_path)
        if not os.path.exists(txt_path):
            higgs_create_voice_reference_txt(ref_audio_path, device)
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content="Please speak this text."),
            Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
            Message(role="user", content=transcript)
        ]
    else: # Smart Voice
        messages = [Message(role="system", content=system_content), Message(role="user", content=transcript)]
        
    output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
    output_path = higgs_get_output_path("basic_generation", "basic_audio")
    torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    gc.collect()
    return output_path

def higgs_run_voice_clone(transcript, uploaded_voice, temperature, max_new_tokens, seed, device):
    load_higgs_model(device)
    if not transcript.strip(): raise gr.Error("Please enter text to synthesize")
    if uploaded_voice is None: raise gr.Error("Please upload a voice sample for cloning")
    if seed > 0: torch.manual_seed(seed)

    temp_audio_path, temp_txt_path = None, None
    try:
        temp_audio_path = higgs_save_temp_audio_fixed(uploaded_voice)
        temp_txt_path = higgs_create_voice_reference_txt(temp_audio_path, device)
        
        system_content = "Generate audio following instruction."
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content="Please speak this text."),
            Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)),
            Message(role="user", content=transcript)
        ]
        
        output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
        output_path = higgs_get_output_path("voice_cloning", "cloned_voice")
        torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        gc.collect()
        return output_path
    finally:
        higgs_robust_file_cleanup([temp_audio_path, temp_txt_path])

def higgs_run_longform(transcript, voice_choice, uploaded_voice, voice_prompt, temperature, max_new_tokens, seed, scene_description, chunk_size, device, progress=gr.Progress()):
    load_higgs_model(device)
    if seed > 0: torch.manual_seed(seed)
    
    chunks = higgs_smart_chunk_text(transcript, max_chunk_size=chunk_size)
    
    temp_audio_path, temp_txt_path, first_chunk_audio_path = None, None, None
    voice_ref_path, voice_ref_text = None, None
    
    try:
        # Determine initial voice reference
        if voice_choice == "Upload Voice" and uploaded_voice is not None:
            temp_audio_path = higgs_save_temp_audio_fixed(uploaded_voice)
            temp_txt_path = higgs_create_voice_reference_txt(temp_audio_path, device)
            voice_ref_path = temp_audio_path
            with open(temp_txt_path, 'r', encoding='utf-8') as f: voice_ref_text = f.read().strip()
        elif voice_choice == "Predefined Voice" and voice_prompt != "None (Smart Voice)":
            voice_ref_path = higgs_get_voice_path(voice_prompt)
            txt_path = higgs_robust_txt_path_creation(voice_ref_path)
            if not os.path.exists(txt_path): higgs_create_voice_reference_txt(voice_ref_path, device)
            with open(txt_path, 'r', encoding='utf-8') as f: voice_ref_text = f.read().strip()

        system_content = "Generate audio following instruction."
        if scene_description: system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        full_audio = []
        sampling_rate = 24000
        
        for i, chunk in enumerate(progress.tqdm(chunks, desc="Generating Chunks")):
            if voice_ref_path: # Uploaded or Predefined
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=voice_ref_text),
                    Message(role="assistant", content=AudioContent(audio_url=voice_ref_path)),
                    Message(role="user", content=chunk)
                ]
            else: # Smart Voice
                if i == 0:
                    messages = [Message(role="system", content=system_content), Message(role="user", content=chunk)]
                else:
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=chunks[0]), # Use first chunk text as prompt
                        Message(role="assistant", content=AudioContent(audio_url=first_chunk_audio_path)),
                        Message(role="user", content=chunk)
                    ]

            output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
            
            if voice_choice == "Smart Voice" and i == 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    first_chunk_audio_path = tmpfile.name
                torchaudio.save(first_chunk_audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)

            full_audio.append(output.audio)
            sampling_rate = output.sampling_rate
            
        if full_audio:
            full_audio_np = np.concatenate(full_audio, axis=0)
            output_path = higgs_get_output_path("longform_generation", "longform_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio_np)[None, :], sampling_rate)
            gc.collect()
            return output_path
        return None
    finally:
        higgs_robust_file_cleanup([temp_audio_path, temp_txt_path, first_chunk_audio_path])

def higgs_run_multi_speaker(transcript, voice_method, audios, voices, temperature, max_new_tokens, seed, scene_description, auto_format, device, progress=gr.Progress()):
    load_higgs_model(device)
    if seed > 0: torch.manual_seed(seed)
    if auto_format: transcript = higgs_auto_format_multi_speaker(transcript)
    
    lines = [line.strip() for line in transcript.split('\n') if line.strip()]
    if not lines: raise gr.Error("Transcript is empty or contains no speaker tags.")

    voice_refs = {}
    temp_files = []
    
    try:
        # Setup voice references
        if voice_method == "Upload Voices":
            for i, audio in enumerate(audios):
                if audio:
                    speaker_key = f"SPEAKER{i}"
                    temp_path = higgs_save_temp_audio_fixed(audio)
                    txt_path = higgs_create_voice_reference_txt(temp_path, device)
                    with open(txt_path, 'r', encoding='utf-8') as f: text = f.read().strip()
                    voice_refs[speaker_key] = {"audio": temp_path, "text": text}
                    temp_files.extend([temp_path, txt_path])
        elif voice_method == "Predefined Voices":
            for i, voice_name in enumerate(voices):
                if voice_name and voice_name != "None (Smart Voice)":
                    speaker_key = f"SPEAKER{i}"
                    audio_path = higgs_get_voice_path(voice_name)
                    txt_path = higgs_robust_txt_path_creation(audio_path)
                    if not os.path.exists(txt_path): higgs_create_voice_reference_txt(audio_path, device)
                    with open(txt_path, 'r', encoding='utf-8') as f: text = f.read().strip()
                    voice_refs[speaker_key] = {"audio": audio_path, "text": text}

        system_content = "Generate audio following instruction."
        if scene_description: system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        full_audio = []
        sampling_rate = 24000
        
        for line in progress.tqdm(lines, desc="Generating Dialogue"):
            match = re.match(r'\[(SPEAKER\d+)\]\s*(.*)', line)
            if not match: continue
            
            speaker_id, text_content = match.groups()
            if not text_content: continue

            if speaker_id in voice_refs: # Uploaded or Predefined
                ref = voice_refs[speaker_id]
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=ref["text"]),
                    Message(role="assistant", content=AudioContent(audio_url=ref["audio"])),
                    Message(role="user", content=text_content)
                ]
            else: # Smart Voice
                if speaker_id in voice_refs: # Already generated for this speaker
                    ref = voice_refs[speaker_id]
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=ref["text"]),
                        Message(role="assistant", content=AudioContent(audio_url=ref["audio"])),
                        Message(role="user", content=text_content)
                    ]
                else: # First time for this speaker
                    messages = [Message(role="system", content=system_content), Message(role="user", content=text_content)]

            output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
            
            if voice_method == "Smart Voice" and speaker_id not in voice_refs:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    audio_path = tmp_audio.name
                torchaudio.save(audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                txt_path = higgs_create_voice_reference_txt(audio_path, device, transcript_sample=text_content)
                voice_refs[speaker_id] = {"audio": audio_path, "text": text_content}
                temp_files.extend([audio_path, txt_path])

            full_audio.append(output.audio)
            sampling_rate = output.sampling_rate
            # Add a small pause between speakers
            full_audio.append(np.zeros(int(0.2 * sampling_rate), dtype=np.float32))

        if full_audio:
            full_audio_np = np.concatenate(full_audio, axis=0)
            output_path = higgs_get_output_path("multi_speaker", "multi_speaker_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio_np)[None, :], sampling_rate)
            gc.collect()
            return output_path
        return None
    finally:
        higgs_robust_file_cleanup(temp_files)

def higgs_run_subtitle_generation(subtitle_file, voice_choice, uploaded_voice, voice_prompt, temperature, seed, device, progress=gr.Progress()):
    load_higgs_model(device)
    if subtitle_file is None: raise gr.Error("Please upload a .srt or .vtt file.")
    if seed > 0: torch.manual_seed(seed)

    temp_ref_path, temp_txt_path = None, None
    try:
        subs = pysrt.open(subtitle_file.name, encoding='utf-8')
        
        voice_ref = None
        if voice_choice == "Upload Voice":
            if uploaded_voice is None: raise gr.Error("Please upload a voice sample.")
            temp_ref_path = higgs_save_temp_audio_fixed(uploaded_voice)
            temp_txt_path = higgs_create_voice_reference_txt(temp_ref_path, device)
            with open(temp_txt_path, 'r', encoding='utf-8') as f: text = f.read().strip()
            voice_ref = {"audio": temp_ref_path, "text": text}
        elif voice_choice == "Predefined Voice":
            if not voice_prompt or voice_prompt == "None (Smart Voice)": raise gr.Error("Please select a predefined voice.")
            audio_path = higgs_get_voice_path(voice_prompt)
            txt_path = higgs_robust_txt_path_creation(audio_path)
            if not os.path.exists(txt_path): higgs_create_voice_reference_txt(audio_path, device)
            with open(txt_path, 'r', encoding='utf-8') as f: text = f.read().strip()
            voice_ref = {"audio": audio_path, "text": text}

        audio_segments = []
        last_end_time = pysrt.SubRipTime(0)
        sample_rate = 24000
        system_content = "Generate audio following instruction."

        for sub in progress.tqdm(subs, desc="Generating from Subtitles"):
            gap_seconds = max(0, (sub.start - last_end_time).to_seconds())
            if gap_seconds > 0:
                audio_segments.append(torch.zeros((1, int(gap_seconds * sample_rate))))
            
            text = sub.text.replace('\n', ' ').strip()
            if not text:
                last_end_time = sub.end
                continue

            if voice_ref: # Uploaded or Predefined
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=voice_ref["text"]),
                    Message(role="assistant", content=AudioContent(audio_url=voice_ref["audio"])),
                    Message(role="user", content=text)
                ]
            else: # Smart Voice
                if not voice_ref: # First line
                    messages = [Message(role="system", content=system_content), Message(role="user", content=text)]
                else: # Subsequent lines
                     messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=voice_ref["text"]),
                        Message(role="assistant", content=AudioContent(audio_url=voice_ref["audio"])),
                        Message(role="user", content=text)
                    ]

            output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=2048, temperature=temperature)
            speech_tensor = torch.from_numpy(output.audio)[None, :]
            
            if voice_choice == "Smart Voice" and not voice_ref:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    temp_ref_path = tmp_audio.name
                torchaudio.save(temp_ref_path, speech_tensor, output.sampling_rate)
                temp_txt_path = higgs_create_voice_reference_txt(temp_ref_path, device, transcript_sample=text)
                voice_ref = {"audio": temp_ref_path, "text": text}

            audio_segments.append(speech_tensor)
            last_end_time = sub.end

        if not audio_segments: raise gr.Error("No audio was generated.")
        full_audio_tensor = torch.cat(audio_segments, dim=1)
        output_path = higgs_get_output_path("subtitle_generation", f"{Path(subtitle_file.name).stem}_timed_audio")
        torchaudio.save(output_path, full_audio_tensor, sample_rate)
        gc.collect()
        return output_path
    finally:
        higgs_robust_file_cleanup([temp_ref_path, temp_txt_path])

def higgs_save_voice_to_library(audio_data, voice_name, device):
    if audio_data is None: return "❌ Please upload an audio sample first"
    if not voice_name or not voice_name.strip(): return "❌ Please enter a voice name"
    
    voice_name_sanitized = voice_name.strip().replace(' ', '_')
    voice_path = os.path.join(HIGGS_VOICE_LIBRARY_PATH, f"{voice_name_sanitized}.wav")
    if os.path.exists(voice_path): return f"❌ Voice '{voice_name_sanitized}' already exists."
    
    try:
        temp_path = higgs_save_temp_audio_fixed(audio_data)
        shutil.move(temp_path, voice_path)
        higgs_create_voice_reference_txt(voice_path, device) # Auto-transcribe on save
        return f"✅ Voice '{voice_name_sanitized}' saved to library!"
    except Exception as e:
        return f"❌ Error saving voice: {e}"

def higgs_delete_voice_from_library(voice_name):
    if not voice_name or voice_name == "None": return "❌ Please select a voice to delete"
    voice_path = os.path.join(HIGGS_VOICE_LIBRARY_PATH, f"{voice_name}.wav")
    txt_path = os.path.join(HIGGS_VOICE_LIBRARY_PATH, f"{voice_name}.txt")
    try:
        if os.path.exists(voice_path): os.remove(voice_path)
        if os.path.exists(txt_path): os.remove(txt_path)
        return f"✅ Voice '{voice_name}' deleted."
    except Exception as e:
        return f"❌ Error deleting voice: {e}"


# ========================================================================================
# --- Gradio UI ---
# ========================================================================================

def create_gradio_ui():
    with gr.Blocks(title="Unified TTS & ASR Pipeline") as demo:
        gr.HTML("""<div style="text-align: center; max-width: 800px; margin: 0 auto;"><h1 style="color: #4CAF50;">Unified TTS & ASR Pipeline</h1><p style="font-size: 1.1em;">A complete toolkit for audio transcription and voiceover generation with Coqui XTTS and Higgs TTS.</p></div>""")
        
        with gr.Accordion("⚙️ Global Device & Process Settings", open=False):
            with gr.Row():
                # This device setting will be used by all models
                tts_device = gr.Radio(label="TTS Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
                whisper_device = gr.Radio(label="Whisper Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
                global_device = gr.Radio(label="Processing Device (requires restart if changed)", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
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
            # --- NEW: Higgs TTS Tab ---
            if HIGGS_AVAILABLE:
                with gr.Tab("Higgs TTS", id=3):
                    with gr.Accordion("Higgs Configuration Management", open=False):
                        with gr.Row():
                            higgs_config_name = gr.Textbox(label="Config Name", placeholder="Enter name to save settings...")
                            higgs_save_config_btn = gr.Button("Save Config")
                        with gr.Row():
                            higgs_load_config_dd = gr.Dropdown(label="Load Config", choices=get_higgs_config_files(), scale=3)
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
                                            higgs_lf_voice_prompt = gr.Dropdown(choices=higgs_get_all_available_voices(), value="None (Smart Voice)", label="Predefined Voice Prompts")
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
                                                    higgs_ms_speaker0_voice = gr.Dropdown(choices=higgs_get_all_available_voices(), label="SPEAKER0 Voice")
                                                    higgs_ms_speaker1_voice = gr.Dropdown(choices=higgs_get_all_available_voices(), label="SPEAKER1 Voice")
                                                    higgs_ms_speaker2_voice = gr.Dropdown(choices=higgs_get_all_available_voices(), label="SPEAKER2 Voice")
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
                                                    higgs_sub_voice_prompt = gr.Dropdown(choices=higgs_get_all_available_voices(), label="Predefined Voice")
                                                    higgs_sub_refresh_voices_2 = gr.Button("Refresh Voice List")
                                            higgs_sub_generate_btn = gr.Button("Generate Timed Audio", variant="primary")
                                        with gr.Column():
                                            higgs_sub_output_audio = gr.Audio(label="Generated Timed Audio", type="filepath")

                        with gr.TabItem("Voice Library"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("### Add New Voice to Library")
                                    higgs_vl_new_voice_audio = gr.Audio(label="Upload Voice Sample", type="numpy")
                                    higgs_vl_new_voice_name = gr.Textbox(label="Voice Name", placeholder="Enter a unique name...")
                                    higgs_vl_save_btn = gr.Button("Save to Library", variant="primary")
                                    higgs_vl_save_status = gr.Textbox(label="Status", interactive=False)
                                with gr.Column():
                                    gr.Markdown("### Manage Existing Voices")
                                    higgs_vl_existing_voices = gr.Dropdown(label="Select Voice to Delete", choices=["None"] + higgs_get_voice_library_voices())
                                    higgs_vl_delete_btn = gr.Button("Delete Selected", variant="stop")
                                    higgs_vl_delete_status = gr.Textbox(label="Delete Status", interactive=False)
                                    higgs_vl_refresh_btn = gr.Button("Refresh Library")
        
        # --- Event Handling ---
        
        # Global
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

        # Higgs Event Handlers
        if HIGGS_AVAILABLE:
            # Configs
            higgs_save_config_btn.click(fn=save_higgs_config, inputs=[higgs_config_name, higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, higgs_lf_scene_description, higgs_lf_chunk_size, higgs_ms_auto_format], outputs=higgs_config_save_status).then(lambda: gr.update(choices=get_higgs_config_files()), None, higgs_load_config_dd)
            higgs_load_config_btn.click(fn=load_higgs_config, inputs=higgs_load_config_dd, outputs=[higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, higgs_lf_scene_description, higgs_lf_chunk_size, higgs_ms_auto_format])
            higgs_delete_config_btn.click(fn=delete_higgs_config, inputs=higgs_load_config_dd, outputs=higgs_config_save_status).then(lambda: gr.update(choices=get_higgs_config_files()), None, higgs_load_config_dd)
            higgs_refresh_configs_btn.click(lambda: gr.update(choices=get_higgs_config_files()), None, higgs_load_config_dd)

            # Basic/Long-form
            higgs_lf_voice_choice.change(lambda choice: {higgs_lf_upload_group: gr.update(visible=choice == "Upload Voice"), higgs_lf_predefined_group: gr.update(visible=choice == "Predefined Voice")}, higgs_lf_voice_choice, [higgs_lf_upload_group, higgs_lf_predefined_group])
            higgs_lf_generate_btn.click(fn=higgs_run_longform, inputs=[higgs_lf_transcript, higgs_lf_voice_choice, higgs_lf_uploaded_voice, higgs_lf_voice_prompt, higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, higgs_lf_scene_description, higgs_lf_chunk_size, global_device], outputs=higgs_lf_output_audio)
            higgs_lf_refresh_voices.click(lambda: gr.update(choices=higgs_get_all_available_voices()), None, higgs_lf_voice_prompt)

            # Voice Cloning
            higgs_vc_generate_btn.click(fn=higgs_run_voice_clone, inputs=[higgs_vc_transcript, higgs_vc_uploaded_voice, higgs_vc_temperature, higgs_vc_max_new_tokens, higgs_vc_seed, global_device], outputs=higgs_vc_output_audio)

            # Multi-speaker
            higgs_ms_voice_method.change(lambda choice: {higgs_ms_upload_group: gr.update(visible=choice == "Upload Voices"), higgs_ms_predefined_group: gr.update(visible=choice == "Predefined Voices")}, higgs_ms_voice_method, [higgs_ms_upload_group, higgs_ms_predefined_group])
            higgs_ms_generate_btn.click(fn=higgs_run_multi_speaker, inputs=[higgs_ms_transcript, higgs_ms_voice_method, [higgs_ms_speaker0_audio, higgs_ms_speaker1_audio, higgs_ms_speaker2_audio], [higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice], higgs_lf_temperature, higgs_lf_max_new_tokens, higgs_lf_seed, higgs_lf_scene_description, higgs_ms_auto_format, global_device], outputs=higgs_ms_output_audio)
            def refresh_multi_voice():
                choices = higgs_get_all_available_voices()
                return gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)
            higgs_ms_refresh_voices_multi.click(refresh_multi_voice, None, [higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice])
            
            # Subtitles
            higgs_sub_voice_choice.change(lambda choice: {higgs_sub_upload_group: gr.update(visible=choice == "Upload Voice"), higgs_sub_predefined_group: gr.update(visible=choice == "Predefined Voice")}, higgs_sub_voice_choice, [higgs_sub_upload_group, higgs_sub_predefined_group])
            higgs_sub_generate_btn.click(fn=higgs_run_subtitle_generation, inputs=[higgs_sub_file_upload, higgs_sub_voice_choice, higgs_sub_uploaded_voice, higgs_sub_voice_prompt, higgs_lf_temperature, higgs_lf_seed, global_device], outputs=higgs_sub_output_audio)
            higgs_sub_refresh_voices_2.click(lambda: gr.update(choices=higgs_get_all_available_voices()), None, higgs_sub_voice_prompt)

            # Voice Library
            def refresh_higgs_library_and_prompts():
                lib_voices = ["None"] + higgs_get_voice_library_voices()
                prompt_voices = higgs_get_all_available_voices()
                return gr.update(choices=lib_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices), gr.update(choices=prompt_voices)
            higgs_vl_save_btn.click(fn=higgs_save_voice_to_library, inputs=[higgs_vl_new_voice_audio, higgs_vl_new_voice_name, global_device], outputs=higgs_vl_save_status).then(refresh_higgs_library_and_prompts, None, [higgs_vl_existing_voices, higgs_lf_voice_prompt, higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice, higgs_sub_voice_prompt])
            higgs_vl_delete_btn.click(fn=higgs_delete_voice_from_library, inputs=higgs_vl_existing_voices, outputs=higgs_vl_delete_status).then(refresh_higgs_library_and_prompts, None, [higgs_vl_existing_voices, higgs_lf_voice_prompt, higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice, higgs_sub_voice_prompt])
            higgs_vl_refresh_btn.click(refresh_higgs_library_and_prompts, None, [higgs_vl_existing_voices, higgs_lf_voice_prompt, higgs_ms_speaker0_voice, higgs_ms_speaker1_voice, higgs_ms_speaker2_voice, higgs_sub_voice_prompt])


    return demo

if __name__ == "__main__":
    if not check_ffmpeg():
        # Allow the app to run anyway, but with a warning
        print("\n⚠️ WARNING: FFmpeg not found. Some audio formats may not work correctly.\n")
    
    app = create_gradio_ui()
    
    print("\n✅ Gradio UI created. Launching Web UI...")
    print("➡️ Access the UI by opening the 'Running on local URL' link below in your browser.")
    
    app.launch(share=True)
