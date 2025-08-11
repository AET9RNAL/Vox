import gradio as gr
import srt
import torch
import torchaudio
import os
import numpy as np
import soundfile as sf
import warnings
import re
from num2words import num2words
from TTS.api import TTS
from datetime import timedelta
import pooch
import tempfile
import json
import shutil
import traceback
import sys


# Coqui Imports
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Stable-TS Imports
import stable_whisper
import stable_whisper.audio

# Define the path to the local model directory.
# This tells the script to load the model from the 'XTTS_model' folder in your project directory.
LOCAL_XTTS_MODEL_PATH = "XTTS_model"
XTTS_AVAILABLE = False
# Coqui Config
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_NAME_FOR_FILE = "Coqui_XTTSv2"
SAMPLE_RATE = 24000
VOICE_LIBRARY_PATH = "voice_library"
TTS_CONFIG_LIBRARY_PATH = "tts_configs"


# Global variables to hold the models
tts_model = None
current_tts_device = None

# Coqui Stock Voices
STOCK_VOICES = {
    'De': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/de_sample.wav",
    'En': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/en_sample.wav",
    'Es': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/es_sample.wav",
    'Fr': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/fr_sample.wav",
    'Ja': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/ja-sample.wav",
    'Pt': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/pt-sample.wav",
    'Tr': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/tr-sample.wav",
    'Zh-Cn': "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/zh-cn-sample.wav"
}

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
]

os.makedirs(VOICE_LIBRARY_PATH, exist_ok=True)
os.makedirs(TTS_CONFIG_LIBRARY_PATH, exist_ok=True)

def check_xtts_availability():
    """Check if the XTTS model files exist and mark XTTS_AVAILABLE."""
    global XTTS_AVAILABLE
    model_dir = os.path.abspath(LOCAL_XTTS_MODEL_PATH)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isdir(model_dir) and os.path.exists(config_path):
        XTTS_AVAILABLE = True
    else:
        XTTS_AVAILABLE = False
    return XTTS_AVAILABLE

# Coqui helper functions
def clear_tts_cache():
    global tts_model
    try:
        if sys.platform == "win32":
            cache_path = os.path.join(os.getenv("LOCALAPPDATA"), "tts")
        else:
            cache_path = os.path.join(os.path.expanduser("~"), ".local", "share", "tts")

        if os.path.exists(cache_path):
            print(f"🚮 Clearing TTS model cache at: {cache_path}")
            shutil.rmtree(cache_path)
            tts_model = None 
            return "✅ TTS model cache cleared successfully. You may need to restart the app."
        else:
            return "ℹ️ TTS model cache directory not found."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error clearing TTS cache: {e}."

def get_library_voices():
    if not os.path.exists(VOICE_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(VOICE_LIBRARY_PATH) if f.endswith(('.wav', '.mp3'))]

def get_tts_config_files():
    if not os.path.exists(TTS_CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(TTS_CONFIG_LIBRARY_PATH) if f.endswith('.json')]

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



# ========================================================================================
# --- Model Loading Functions ---
# ========================================================================================
def load_tts_model(device):
    """
    *** FIX ***: This function now loads the Coqui TTS model from a local directory
    and applies a workaround for a bug in TTS v0.22.0 that causes a TypeError
    when the speaker_encoder_config_path is null in the config.json.
    """
    global tts_model, current_tts_device, XTTS_AVAILABLE
    if tts_model is not None and current_tts_device == device:
        return "TTS model is ready."


    model_dir = os.path.abspath(LOCAL_XTTS_MODEL_PATH)
    if not os.path.isdir(model_dir):
        error_message = (
            f"❌ Model directory not found at '{model_dir}'.\n"
            f"Please ensure the '{LOCAL_XTTS_MODEL_PATH}' folder exists in the same directory as the script "
            "and contains the downloaded model files."
        )
        raise gr.Error(error_message)

    original_config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(original_config_path):
        error_message = f"❌ 'config.json' not found in the model directory: '{model_dir}'."
        raise gr.Error(error_message)

    # --- Workaround for TypeError ---
    temp_config_path = None
    dummy_se_config_path = None
    config_to_use = original_config_path

    try:
        with open(original_config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        if config_data.get("model_args", {}).get("speaker_encoder_config_path") is None:
            print("ℹ️ Applying workaround for 'speaker_encoder_config_path': null issue.")
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir=model_dir, encoding='utf-8') as dummy_file:
                json.dump({}, dummy_file)
                dummy_se_config_path = dummy_file.name
            
            config_data["model_args"]["speaker_encoder_config_path"] = os.path.basename(dummy_se_config_path)

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_temp_main.json', dir=model_dir, encoding='utf-8') as temp_main_file:
                json.dump(config_data, temp_main_file, indent=2)
                temp_config_path = temp_main_file.name
            
            config_to_use = temp_config_path
        # --- End of Workaround ---

        print(f"⏳ Loading Coqui TTS model from local path: {model_dir} to device: {device}...")
        
        # Load the model using the (potentially temporary) config file.
        # The TTS class will use the config_path and find other model files in the same directory.
        tts_model = TTS(
            model_path=model_dir, 
            config_path=config_to_use, 
            gpu=(device == 'cuda')
        )
        
        current_tts_device = device
        print(f"✅ TTS Model loaded successfully from local files on {device}.")
    
    except Exception as e:
        print("\n--- DETAILED TTS LOADING ERROR ---")
        traceback.print_exc()
        print("----------------------------------\n")
        error_message = f"❌ Failed to load local TTS model: {e}"
        raise gr.Error(error_message)

    
    finally:
        # --- Cleanup temporary files ---
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        if dummy_se_config_path and os.path.exists(dummy_se_config_path):
            os.remove(dummy_se_config_path)
        # --- End of Cleanup ---
            
    return "TTS model is ready."



# ========================================================================================
# --- Core Logic & Helper Functions ---
# ========================================================================================
def normalize_text(text, lang='en'):
    """Normalizes numbers in text to words, aware of the language."""
    try:
        # This regex is language-agnostic for finding numbers
        return re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang=lang), text)
    except NotImplementedError:
        # Fallback for languages not supported by num2words
        print(f"⚠️ num2words does not support language '{lang}'. Numbers will not be converted to words.")
        return text
    except Exception as e:
        print(f"⚠️ Error in text normalization: {e}")
        return text


# -------------------------------
# Smart splitting
# -------------------------------
_ABBREVS = set([
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.",
    "fig.", "no.", "inc.", "ltd.", "dept.", "approx."
])

def _is_bad_period(text, idx):
    """Avoid splitting on abbreviations, initials, or decimals like 3.14."""
    left = text[max(0, idx-6):idx+1].lower()
    # decimal number like 3.14
    if re.search(r"\d\.\d", text[max(0, idx-1):idx+2]): 
        return True
    # single-letter initial "A." or "B."
    if re.search(r"\b[A-Z]\.$", left): 
        return True
    # common abbreviations
    for ab in _ABBREVS:
        if left.endswith(ab): 
            return True
    return False

def _find_split_index(text, soft_limit, hard_limit):
    n = len(text)
    if n <= hard_limit: 
        return None  # no split needed

    window_start = min(soft_limit, n-1)
    window_end = min(hard_limit, n-1)

    # 1) strong punctuation . ! ? …
    strong = [".", "!", "?", "…"]
    for i in range(window_end, window_start-1, -1):
        ch = text[i]
        if ch in strong and not _is_bad_period(text, i):
            return i+1  # keep punctuation on the left

    # 2) secondary punctuation , ; : — – -
    secondary = [",", ";", ":", "—", "–", "-"]
    for i in range(window_end, window_start-1, -1):
        if text[i] in secondary:
            return i+1

    # 3) whitespace
    for i in range(window_end, window_start-1, -1):
        if text[i].isspace():
            return i

    # 4) last resort: hard cut
    return hard_limit

def split_long_text(text, soft_limit=210, hard_limit=240):
    """
    Splits text into chunks using punctuation/space preferences.
    Ensures each chunk <= hard_limit (internally clamped to 248 for XTTS safety).
    """
    hard_cap = min(int(hard_limit), 248)
    soft_cap = min(int(soft_limit), hard_cap-10) if soft_limit else hard_cap-10
    if soft_cap < 50: 
        soft_cap = 50  # sanity floor

    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    chunks = []
    cursor = 0
    while cursor < len(text):
        remaining = text[cursor:]
        if len(remaining) <= hard_cap:
            chunks.append(remaining.strip())
            break
        split_at = _find_split_index(remaining, soft_cap, hard_cap)
        left = remaining[:split_at].strip()
        if not left:
            left = remaining[:hard_cap].strip()
            split_at = len(left)
        chunks.append(left)
        cursor += split_at
    return chunks

def _expand_segments_with_splitter(segments, soft_limit, hard_limit, is_timed):
    """
    Apply splitting to segments:
      - Text File Mode (is_timed=False): split any long line into multiple segments.
      - SRT/VTT Mode (is_timed=True): only split if a single cue exceeds hard_limit; distribute time proportionally.
    """
    expanded = []
    if not segments:
        return expanded

    hard_cap = min(int(hard_limit), 248)
    soft_cap = min(int(soft_limit), hard_cap - 10) if soft_limit else hard_cap - 10

    if not is_timed:
        idx = 1
        for sub in segments:
            parts = split_long_text(sub.content, soft_cap, hard_cap)
            if not parts:
                continue
            for p in parts:
                expanded.append(srt.Subtitle(index=idx, start=timedelta(0), end=timedelta(0), content=p))
                idx += 1
        return expanded

    # Timed mode: distribute durations proportionally
    idx = 1
    for sub in segments:
        text_line = re.sub(r"\s+", " ", sub.content.strip())
        if len(text_line) <= hard_cap:
            expanded.append(srt.Subtitle(index=idx, start=sub.start, end=sub.end, content=text_line))
            idx += 1
            continue
        parts = split_long_text(text_line, soft_cap, hard_cap)
        if not parts:
            continue
        total_chars = sum(len(p) for p in parts) or 1
        dur = (sub.end - sub.start).total_seconds()
        cursor = sub.start
        for k, p in enumerate(parts):
            if k < len(parts) - 1:
                frac = len(p) / total_chars
                sec = max(0.05, dur * frac)
                end = cursor + timedelta(seconds=sec)
            else:
                end = sub.end
            expanded.append(srt.Subtitle(index=idx, start=cursor, end=end, content=p))
            cursor = end
            idx += 1
    return expanded

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
        # FIX: Pass language to normalization function
        text_to_speak = normalize_text(raw_text, lang=language)
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
            raise Exception(f"Error on segment {sub.index}: {e}")
            
    if not timed_generation or (timed_generation and not strict_timing):
        final_audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)
    
    sf.write(output_path, final_audio, sample_rate)
    return output_path
    
def safe_load_audio(audio_file_path):
    try:
        return stable_whisper.audio.load_audio(audio_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")

def find_and_replace(result, find_word, replace_word):
    if not find_word or replace_word is None: return result
    for segment in result.segments:
        for word in segment.words:
            if word.word.strip().lower() == find_word.strip().lower():
                word.word = f" {replace_word} "
    return result

def save_voice_to_library(audio_filepath, voice_name):
    if audio_filepath is None: return "❌ Error: Please upload an audio sample first."
    if not voice_name or not voice_name.strip(): return "❌ Error: Please enter a name for the voice."
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", voice_name).strip().replace(" ", "_")
    if not sanitized_name: return "❌ Error: The provided voice name is invalid after sanitization."
    destination_path = os.path.join(VOICE_LIBRARY_PATH, f"{sanitized_name}.wav")
    if os.path.exists(destination_path): return f"❌ Error: A voice with the name '{sanitized_name}' already exists."
    try:
        shutil.copyfile(audio_filepath, destination_path)
        return f"✅ Voice '{sanitized_name}' saved successfully to the library."
    except Exception as e:
        return f"❌ Error saving voice: {e}"



# ========================================================================================
# Coqui Core

def run_tts_generation(
    input_file, language, voice_mode, clone_source, library_voice, clone_speaker_audio, stock_voice,
    output_format, input_mode, srt_timing_mode, tts_device,
    soft_limit=210, hard_limit=240,
    progress=gr.Progress(track_tqdm=True)
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

        # Apply sentence splitting to keep under model limits
        segments = _expand_segments_with_splitter(segments, soft_limit, hard_limit, is_timed)


        output_dir = "gradio_outputs"
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
        # Ensure Gradio errors are properly propagated to the UI
        if isinstance(e, gr.Error):
            raise e
        return None, f"❌ An unexpected error occurred: {e}"

