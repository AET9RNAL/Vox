import gradio as gr
import torch
import torchaudio
import os
import numpy as np
import warnings
import re
from num2words import num2words
from TTS.api import TTS
from datetime import timedelta, datetime
import tempfile
import whisper
import json
import shutil
import traceback
import gc
from pathlib import Path
import pysrt








# Higgs Audio Imports with enhanced path handling
HIGGS_AVAILABLE = False
higgs_import_error = None
SAMPLE_RATE = 24000
#Declare Higgs components globally
#These will be populated by the try_import_higgs function.
HiggsAudioServeEngine = None
HiggsAudioResponse = None
ChatMLSample = None
Message = None
AudioContent = None


# Higgs Audio Import Function
def try_import_higgs():
    """Try to import Higgs Audio with multiple fallback methods and enhanced diagnostics."""
    global HIGGS_AVAILABLE, higgs_import_error
    # Make sure we are modifying the global variables ---
    global HiggsAudioServeEngine, HiggsAudioResponse, ChatMLSample, Message, AudioContent
    
    print("🔍 Attempting to import Higgs Audio...")
    
    # Method 1: Try direct import (if already installed)
    try:
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine as HASE, HiggsAudioResponse as HAR
        from boson_multimodal.data_types import ChatMLSample as CMS, Message as M, AudioContent as AC
        
        # Assign to global variables
        HiggsAudioServeEngine, HiggsAudioResponse, ChatMLSample, Message, AudioContent = HASE, HAR, CMS, M, AC
        
        HIGGS_AVAILABLE = True
        print("✅ Higgs Audio library imported successfully (direct import)")
        return True
    except ImportError as e:
        higgs_import_error = str(e)
        print(f"⚠️ Direct import failed: {e}")
        
        # Provide specific guidance based on error type
        if "AutoProcessor" in str(e):
            print("💡 DIAGNOSIS: Transformers version incompatibility detected")
            print("💡 SOLUTION: Run 'fix_higgs_dependencies.bat' to fix version conflicts")
        elif "torchvision::nms" in str(e):
            print("💡 DIAGNOSIS: PyTorch/TorchVision compatibility issue detected")
            print("💡 SOLUTION: Run 'fix_higgs_dependencies.bat' to fix PyTorch versions")
    except Exception as e:
        higgs_import_error = str(e)
        print(f"⚠️ Unexpected error in direct import: {e}")
        if "torchvision::nms" in str(e):
            print("💡 DIAGNOSIS: PyTorch/TorchVision NMS operation compatibility issue")
            print("💡 SOLUTION: Run 'fix_higgs_dependencies.bat' to fix PyTorch versions")
    
    # Method 2: Try adding higgs-audio directory to path
    try:
        import sys
        import os
        higgs_path = os.path.join(os.path.dirname(__file__), "higgs-audio")
        if os.path.exists(higgs_path) and higgs_path not in sys.path:
            print(f"📁 Adding higgs-audio path: {higgs_path}")
            sys.path.insert(0, higgs_path)
            
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine as HASE, HiggsAudioResponse as HAR
            from boson_multimodal.data_types import ChatMLSample as CMS, Message as M, AudioContent as AC
            
            # Assign to global variables
            HiggsAudioServeEngine, HiggsAudioResponse, ChatMLSample, Message, AudioContent = HASE, HAR, CMS, M, AC

            HIGGS_AVAILABLE = True
            print("✅ Higgs Audio library imported successfully (with path modification)")
            return True
    except ImportError as e:
        print(f"⚠️ Path-based import failed: {e}")
        higgs_import_error = str(e)
        
        # Enhanced error diagnosis
        if "AutoProcessor" in str(e):
            print("💡 DIAGNOSIS: Transformers version is incompatible with Higgs Audio")
            try:
                import transformers
                print(f"💡 Current transformers version: {transformers.__version__}")
                print("💡 Required: 4.45.1 <= version < 4.47.0")
            except:
                pass
        elif "No module named" in str(e):
            print("💡 DIAGNOSIS: Missing Higgs Audio installation")
            print("💡 SOLUTION: Run 'pip install -e ./higgs-audio' or use setup-run.bat")
    except Exception as e:
        print(f"⚠️ Unexpected error during path-based import: {e}")
        higgs_import_error = str(e)
    
    # Method 3: Enhanced directory check and diagnostics
    higgs_dir = os.path.join(os.path.dirname(__file__), "higgs-audio")
    if os.path.exists(higgs_dir):
        print(f"📁 Higgs-audio directory found at: {higgs_dir}")
        boson_dir = os.path.join(higgs_dir, "boson_multimodal")
        if os.path.exists(boson_dir):
            print(f"📁 boson_multimodal package found at: {boson_dir}")
            print("💡 Try running: conda run -n coqui_tts_env pip install -e ./higgs-audio")
        else:
            print("❌ boson_multimodal package not found in higgs-audio directory")
            print("💡 SOLUTION: Re-clone the repository or run setup-run.bat")
    else:
        print("❌ higgs-audio directory not found")
    
    print("⚠️ Higgs Audio library (boson_multimodal) not available.")
    print("⚠️ The 'Higgs TTS' tab will be disabled.")
    print(f"⚠️ Last error: {higgs_import_error}")
    
    # Run detailed verification
    verify_higgs_installation()
    
    return False

def verify_higgs_installation():
    """Verify Higgs Audio installation and provide detailed diagnostics."""
    global HIGGS_AVAILABLE, higgs_import_error
    
    if HIGGS_AVAILABLE:
        return True
    
    print("\n=== HIGGS AUDIO INSTALLATION VERIFICATION ===")
    
    # Check if higgs-audio directory exists
    higgs_dir = os.path.join(os.path.dirname(__file__), "higgs-audio")
    if not os.path.exists(higgs_dir):
        print("❌ higgs-audio directory not found")
        return False
    
    print(f"✅ higgs-audio directory found at: {higgs_dir}")
    
    # Check if boson_multimodal exists
    boson_dir = os.path.join(higgs_dir, "boson_multimodal")
    if not os.path.exists(boson_dir):
        print("❌ boson_multimodal package not found")
        return False
    
    print(f"✅ boson_multimodal package found at: {boson_dir}")
    
    # Check if serve module exists
    serve_dir = os.path.join(boson_dir, "serve")
    if not os.path.exists(serve_dir):
        print("❌ serve module not found in boson_multimodal")
        return False
    
    print("✅ serve module found")
    
    # Check if __init__.py exists in serve directory
    serve_init = os.path.join(serve_dir, "__init__.py")
    if not os.path.exists(serve_init):
        print("❌ __init__.py not found in serve module")
        return False
    
    print("✅ serve module has __init__.py")
    
    # Check if serve_engine.py exists
    serve_engine = os.path.join(serve_dir, "serve_engine.py")
    if not os.path.exists(serve_engine):
        print("❌ serve_engine.py not found")
        return False
    
    print("✅ serve_engine.py found")
    
    # Check transformers version
    try:
        import transformers
        version = transformers.__version__
        print(f"✅ Transformers version: {version}")
        
        # Check if version is compatible
        version_parts = version.split('.')
        major_minor = float(f"{version_parts[0]}.{version_parts[1]}")
        if not (4.45 <= major_minor < 4.47):
            print(f"⚠️ WARNING: Transformers version {version} may not be compatible")
            print("   Required: 4.45.1 <= version < 4.47.0")
    except Exception as e:
        print(f"⚠️ WARNING: Could not check Transformers version: {e}")
    
    # Try to check if the package is installed in the environment
    try:
        import subprocess
        import sys
        
        # Get the current environment's site-packages
        result = subprocess.run([sys.executable, "-c", "import site; print(site.getsitepackages()[0])"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            site_packages = result.stdout.strip()
            print(f"✅ Site packages directory: {site_packages}")
            
            # Check if boson_multimodal is in site-packages
            boson_in_site = os.path.join(site_packages, "boson_multimodal")
            if os.path.exists(boson_in_site):
                print("✅ boson_multimodal found in site-packages")
            else:
                print("❌ boson_multimodal not found in site-packages")
                print("💡 This suggests the package wasn't installed correctly")
                return False
    except Exception as e:
        print(f"⚠️ WARNING: Could not check site-packages: {e}")
    
    print("=== END VERIFICATION ===\n")
    return False


# Faster Whisper for Higgs
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("✅ Using faster-whisper for Higgs transcription")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("⚠️ faster-whisper not available for Higgs - voice samples will use dummy text")


# Higgs Config
HIGGS_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
HIGGS_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
HIGGS_CONFIG_LIBRARY_PATH = "higgs_configs"
HIGGS_VOICE_LIBRARY_PATH = "higgs_voice_library"

# Global variables to hold the models
higgs_serve_engine = None # Higgs Model
higgs_whisper_model = None # Separate whisper for Higgs if needed
current_higgs_device = None


# ========================================================================================
# --- Library and Config Functions ---
# ========================================================================================
os.makedirs(HIGGS_CONFIG_LIBRARY_PATH, exist_ok=True)
os.makedirs(HIGGS_VOICE_LIBRARY_PATH, exist_ok=True)
os.makedirs("higgs_outputs/basic_generation", exist_ok=True)
os.makedirs("higgs_outputs/voice_cloning", exist_ok=True)
os.makedirs("higgs_outputs/longform_generation", exist_ok=True)
os.makedirs("higgs_outputs/multi_speaker", exist_ok=True)
os.makedirs("higgs_outputs/subtitle_generation", exist_ok=True)


# ======================================================================================== 
# Higgs Configs
def get_higgs_config_files():
    if not os.path.exists(HIGGS_CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(HIGGS_CONFIG_LIBRARY_PATH) if f.endswith('.json')]

def save_higgs_config(config_name, temperature, max_new_tokens, seed, scene_description, chunk_size, auto_format):
    if not config_name or not config_name.strip():
        return "❌ Error: Please enter a name for the configuration."
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", config_name).strip().replace(" ", "_")
    config_path = os.path.join(HIGGS_CONFIG_LIBRARY_PATH, f"{sanitized_name}.json")
    config_data = {
        "temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed,
        "scene_description": scene_description, "chunk_size": chunk_size, "auto_format": auto_format,
    }
    with open(config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=4)
    return f"✅ Higgs Config '{sanitized_name}' saved."

def load_higgs_config(config_name):
    if not config_name: return [gr.update()]*6
    config_path = os.path.join(HIGGS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path): return [gr.update()]*6
    with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
    return [
        gr.update(value=config_data.get("temperature")), gr.update(value=config_data.get("max_new_tokens")),
        gr.update(value=config_data.get("seed")), gr.update(value=config_data.get("scene_description")),
        gr.update(value=config_data.get("chunk_size")), gr.update(value=config_data.get("auto_format")),
    ]

def delete_higgs_config(config_name):
    if not config_name: return "ℹ️ No config selected to delete."
    config_path = os.path.join(HIGGS_CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if os.path.exists(config_path):
        os.remove(config_path)
        return f"✅ Higgs Config '{config_name}' deleted."
    return f"❌ Error: Config '{config_name}' not found."


# ========================================================================================
# Higgs Loader
def load_higgs_model(device):
    global higgs_serve_engine, current_higgs_device
    if not HIGGS_AVAILABLE: raise gr.Error("Higgs Audio library not installed. Please run setup-run.bat.")
    if higgs_serve_engine is None or current_higgs_device != device:
        print(f"⏳ Loading Higgs Audio model to device: {device}...")
        try:
            # Ensure HiggsAudioServeEngine is not None before using it ---
            if HiggsAudioServeEngine is None:
                raise RuntimeError("HiggsAudioServeEngine not imported. Check installation.")
            higgs_serve_engine = HiggsAudioServeEngine(HIGGS_MODEL_PATH, HIGGS_AUDIO_TOKENIZER_PATH, device=device)
            current_higgs_device = device
            print(f"✅ Higgs Audio Model loaded successfully on {device}.")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Higgs Audio model: {e}")
    return "Higgs Audio model is ready."

def load_higgs_whisper_model(device):
    global higgs_whisper_model
    if not FASTER_WHISPER_AVAILABLE: return
    if higgs_whisper_model is None:
        print("⏳ Loading faster-whisper model for Higgs...")
        try:
            higgs_whisper_model = WhisperModel("base", device=device, compute_type="float16" if device == "cuda" else "int8")
            print("✅ faster-whisper model loaded.")
        except Exception as e:
            print(f"❌ Failed to load faster-whisper model: {e}")


# ========================================================================================
# ... Higgs Helper Functions ...
def higgs_save_temp_audio_robust(audio_data, sample_rate):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    if isinstance(audio_data, np.ndarray):
        waveform = torch.from_numpy(audio_data).float()
        if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
        torchaudio.save(temp_path, waveform, sample_rate)
    return temp_path

def higgs_process_uploaded_audio(uploaded_audio):
    if uploaded_audio is None: return None, None
    sample_rate, audio_data = uploaded_audio
    if isinstance(audio_data, np.ndarray):
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16: audio_data = audio_data.astype(np.float32) / 32768.0
            else: audio_data = audio_data.astype(np.float32)
        if len(audio_data.shape) > 1: audio_data = np.mean(audio_data, axis=1)
        return audio_data, sample_rate
    return None, None

def higgs_save_temp_audio_fixed(uploaded_voice):
    if uploaded_voice is None: return None
    processed_audio, processed_rate = higgs_process_uploaded_audio(uploaded_voice)
    if processed_audio is not None: return higgs_save_temp_audio_robust(processed_audio, processed_rate)
    return None

def higgs_transcribe_audio(audio_path, device):
    if not FASTER_WHISPER_AVAILABLE: return "This is a voice sample for cloning."
    try:
        load_higgs_whisper_model(device)
        if higgs_whisper_model is None: return "This is a voice sample for cloning."
        segments, _ = higgs_whisper_model.transcribe(audio_path, language="en")
        transcription = " ".join([segment.text for segment in segments]).strip()
        if not transcription: transcription = "This is a voice sample for cloning."
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
    with open(txt_path, 'w', encoding='utf-8') as f: f.write(transcript_sample)
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
            if f.endswith('.wav'): voices.append(os.path.splitext(f)[0])
    return voices

def higgs_get_all_available_voices():
    library = higgs_get_voice_library_voices()
    combined = ["None (Smart Voice)"]
    if library: combined.extend([f"👤 {voice}" for voice in library])
    return combined

def higgs_get_voice_path(voice_selection):
    if not voice_selection or voice_selection == "None (Smart Voice)": return None
    if voice_selection.startswith("👤 "):
        voice_name = voice_selection[2:]
        return os.path.join(HIGGS_VOICE_LIBRARY_PATH, f"{voice_name}.wav")
    return None

def higgs_smart_chunk_text(text, max_chunk_size=200):
    paragraphs = text.split('\n\n')
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        if len(p) <= max_chunk_size:
            chunks.append(p)
            continue
        sentences = re.split(r'(?<=[.!?])\s+', p)
        current_chunk = ""
        for s in sentences:
            if len(current_chunk) + len(s) + 1 > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = s
            else:
                current_chunk += (" " if current_chunk else "") + s
        if current_chunk: chunks.append(current_chunk)
    return chunks

def higgs_parse_multi_speaker_text(text):
    speaker_pattern = r'\[SPEAKER(\d+)\]\s*([^[]*?)(?=\[SPEAKER\d+\]|$)'
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    speakers = {}
    for speaker_id, content in matches:
        speaker_key = f"SPEAKER{speaker_id}"
        if speaker_key not in speakers: speakers[speaker_key] = []
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
# Higgs Core
def higgs_run_basic_generation(transcript, voice_prompt, temperature, max_new_tokens, seed, scene_description, device):
    load_higgs_model(device)
    if seed > 0: torch.manual_seed(seed)
    system_content = "Generate audio following instruction."
    if scene_description: system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
    ref_audio_path = higgs_get_voice_path(voice_prompt)
    if ref_audio_path and os.path.exists(ref_audio_path):
        txt_path = higgs_robust_txt_path_creation(ref_audio_path)
        if not os.path.exists(txt_path): higgs_create_voice_reference_txt(ref_audio_path, device)
        messages = [ Message(role="system", content=system_content), Message(role="user", content="Please speak this text."), Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)), Message(role="user", content=transcript) ]
    else:
        messages = [Message(role="system", content=system_content), Message(role="user", content=transcript)]
    output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
    output_path = higgs_get_output_path("basic_generation", "basic_audio")
    torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    gc.collect(); torch.cuda.empty_cache()
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
        messages = [ Message(role="system", content=system_content), Message(role="user", content="Please speak this text."), Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)), Message(role="user", content=transcript) ]
        output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
        output_path = higgs_get_output_path("voice_cloning", "cloned_voice")
        torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        gc.collect(); torch.cuda.empty_cache()
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
            if voice_ref_path:
                messages = [ Message(role="system", content=system_content), Message(role="user", content=voice_ref_text), Message(role="assistant", content=AudioContent(audio_url=voice_ref_path)), Message(role="user", content=chunk) ]
            else:
                if i == 0: messages = [Message(role="system", content=system_content), Message(role="user", content=chunk)]
                else: messages = [ Message(role="system", content=system_content), Message(role="user", content=chunks[0]), Message(role="assistant", content=AudioContent(audio_url=first_chunk_audio_path)), Message(role="user", content=chunk) ]
            output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
            if voice_choice == "Smart Voice" and i == 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile: first_chunk_audio_path = tmpfile.name
                torchaudio.save(first_chunk_audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
            full_audio.append(output.audio)
            sampling_rate = output.sampling_rate
        if full_audio:
            full_audio_np = np.concatenate(full_audio, axis=0)
            output_path = higgs_get_output_path("longform_generation", "longform_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio_np)[None, :], sampling_rate)
            gc.collect(); torch.cuda.empty_cache()
            return output_path
        return None
    finally:
        higgs_robust_file_cleanup([temp_audio_path, temp_txt_path, first_chunk_audio_path])

def higgs_run_multi_speaker(transcript, voice_method, speaker0_audio, speaker1_audio, speaker2_audio, speaker0_voice, speaker1_voice, speaker2_voice, temperature, max_new_tokens, seed, scene_description, auto_format, device, progress=gr.Progress()):
    # This function combines the individual speaker inputs back into lists.
    audios = [speaker0_audio, speaker1_audio, speaker2_audio]
    voices = [speaker0_voice, speaker1_voice, speaker2_voice]

    load_higgs_model(device)
    if seed > 0: torch.manual_seed(seed)
    if auto_format: transcript = higgs_auto_format_multi_speaker(transcript)
    lines = [line.strip() for line in transcript.split('\n') if line.strip()]
    if not lines: raise gr.Error("Transcript is empty or contains no speaker tags.")
    voice_refs, temp_files = {}, []
    try:
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
        full_audio, sampling_rate = [], 24000
        for line in progress.tqdm(lines, desc="Generating Dialogue"):
            match = re.match(r'\[(SPEAKER\d+)\]\s*(.*)', line)
            if not match: continue
            speaker_id, text_content = match.groups()
            if not text_content: continue
            if speaker_id in voice_refs:
                ref = voice_refs[speaker_id]
                messages = [ Message(role="system", content=system_content), Message(role="user", content=ref["text"]), Message(role="assistant", content=AudioContent(audio_url=ref["audio"])), Message(role="user", content=text_content) ]
            else:
                messages = [Message(role="system", content=system_content), Message(role="user", content=text_content)]
            output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=max_new_tokens, temperature=temperature)
            if voice_method == "Smart Voice" and speaker_id not in voice_refs:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio: audio_path = tmp_audio.name
                torchaudio.save(audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                txt_path = higgs_create_voice_reference_txt(audio_path, device, transcript_sample=text_content)
                voice_refs[speaker_id] = {"audio": audio_path, "text": text_content}
                temp_files.extend([audio_path, txt_path])
            full_audio.append(output.audio)
            sampling_rate = output.sampling_rate
            full_audio.append(np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32))
        if full_audio:
            full_audio_np = np.concatenate(full_audio, axis=0)
            output_path = higgs_get_output_path("multi_speaker", "multi_speaker_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio_np)[None, :], sampling_rate)
            gc.collect(); torch.cuda.empty_cache()
            return output_path
        return None
    finally:
        higgs_robust_file_cleanup(temp_files)

def higgs_run_subtitle_generation(subtitle_file, voice_choice, uploaded_voice, voice_prompt, temperature, seed, timing_mode, device, progress=gr.Progress()):
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
        
        sample_rate = 24000
        is_strict = timing_mode == "Strict (Cut audio to fit)"
        
        if is_strict:
            total_duration_sub = subs[-1].end
            total_duration_seconds = (total_duration_sub.hours * 3600) + (total_duration_sub.minutes * 60) + total_duration_sub.seconds + (total_duration_sub.milliseconds / 1000)
            final_audio = torch.zeros((1, int(total_duration_seconds * sample_rate)))
        else:
            audio_segments = []
            last_end_time = pysrt.SubRipTime(0)

        system_content = "Generate audio following instruction."
        for sub in progress.tqdm(subs, desc="Generating from Subtitles"):
            text = sub.text.replace('\n', ' ').strip()
            if not text:
                if not is_strict:
                    last_end_time = sub.end
                continue

            if voice_ref:
                messages = [ Message(role="system", content=system_content), Message(role="user", content=voice_ref["text"]), Message(role="assistant", content=AudioContent(audio_url=voice_ref["audio"])), Message(role="user", content=text) ]
            else:
                messages = [Message(role="system", content=system_content), Message(role="user", content=text)]
            
            output: HiggsAudioResponse = higgs_serve_engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=2048, temperature=temperature)
            speech_tensor = torch.from_numpy(output.audio)[None, :]
            
            if voice_choice == "Smart Voice" and not voice_ref:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio: temp_ref_path = tmp_audio.name
                torchaudio.save(temp_ref_path, speech_tensor, output.sampling_rate)
                temp_txt_path = higgs_create_voice_reference_txt(temp_ref_path, device, transcript_sample=text)
                voice_ref = {"audio": temp_ref_path, "text": text}

            if is_strict:
                start_time_sub = sub.start
                start_time_sec = (start_time_sub.hours * 3600) + (start_time_sub.minutes * 60) + start_time_sub.seconds + (start_time_sub.milliseconds / 1000)
                
                end_time_sub = sub.end
                time_diff_sub = end_time_sub - start_time_sub
                subtitle_duration_sec = (time_diff_sub.hours * 3600) + (time_diff_sub.minutes * 60) + time_diff_sub.seconds + (time_diff_sub.milliseconds / 1000)

                generated_duration_sec = speech_tensor.shape[1] / sample_rate
                if generated_duration_sec > subtitle_duration_sec:
                    warnings.warn(f"\nLine {sub.index}: Speech ({generated_duration_sec:.2f}s) is LONGER than subtitle duration ({subtitle_duration_sec:.2f}s) and will be CUT OFF.")

                start_sample = int(start_time_sec * sample_rate)
                end_sample = start_sample + speech_tensor.shape[1]
                
                if end_sample > final_audio.shape[1]:
                    speech_tensor = speech_tensor[:, :final_audio.shape[1] - start_sample]
                    end_sample = final_audio.shape[1]
                
                final_audio[:, start_sample:end_sample] = speech_tensor
            else: # Flexible
                time_diff = sub.start - last_end_time
                gap_seconds = max(0, time_diff.hours * 3600 + time_diff.minutes * 60 + time_diff.seconds + time_diff.milliseconds / 1000)
                if gap_seconds > 0: audio_segments.append(torch.zeros((1, int(gap_seconds * sample_rate))))
                
                audio_segments.append(speech_tensor)
                last_end_time = sub.end
        
        if is_strict:
            full_audio_tensor = final_audio
        else:
            if not audio_segments: raise gr.Error("No audio was generated.")
            full_audio_tensor = torch.cat(audio_segments, dim=1)

        output_path = higgs_get_output_path("subtitle_generation", f"{Path(subtitle_file.name).stem}_timed_audio")
        torchaudio.save(output_path, full_audio_tensor, sample_rate)
        gc.collect(); torch.cuda.empty_cache()
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
        higgs_create_voice_reference_txt(voice_path, device)
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
