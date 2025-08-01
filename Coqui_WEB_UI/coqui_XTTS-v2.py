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

# ✨ Import the necessary config classes for PyTorch's safelist
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# ========================================================================================
# --- Global Model and Configuration ---
# ========================================================================================

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_NAME_FOR_FILE = "Coqui_XTTSv2"
SAMPLE_RATE = 24000
VOICE_LIBRARY_PATH = "voice_library"
CONFIG_LIBRARY_PATH = "tts_configs"

# Global variables to hold the models
tts_model = None
whisper_model = None
current_tts_device = None
current_whisper_device = None
current_whisper_model_size = None

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
# --- Library and Config Functions ---
# ========================================================================================
os.makedirs(VOICE_LIBRARY_PATH, exist_ok=True)
os.makedirs(CONFIG_LIBRARY_PATH, exist_ok=True)

def get_library_voices():
    if not os.path.exists(VOICE_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(VOICE_LIBRARY_PATH) if f.endswith(('.wav', '.mp3'))]

def get_config_files():
    if not os.path.exists(CONFIG_LIBRARY_PATH): return []
    return [os.path.splitext(f)[0] for f in os.listdir(CONFIG_LIBRARY_PATH) if f.endswith('.json')]

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

def save_tts_config(config_name, language, voice_mode, clone_source, library_voice, stock_voice, output_format, srt_timing_mode):
    if not config_name or not config_name.strip():
        return "❌ Error: Please enter a name for the configuration."
    
    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", config_name).strip().replace(" ", "_")
    if not sanitized_name:
        return "❌ Error: Invalid config name."

    config_path = os.path.join(CONFIG_LIBRARY_PATH, f"{sanitized_name}.json")
    
    config_data = {
        "language": language,
        "voice_mode": voice_mode,
        "clone_source": clone_source,
        "library_voice": library_voice,
        "stock_voice": stock_voice,
        "output_format": output_format,
        "srt_timing_mode": srt_timing_mode
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        return f"✅ Config '{sanitized_name}' saved successfully."
    except Exception as e:
        return f"❌ Error saving config: {e}"

def load_tts_config(config_name):
    if not config_name:
        return [gr.update()]*7 # Return updates for all 7 fields
    
    config_path = os.path.join(CONFIG_LIBRARY_PATH, f"{config_name}.json")
    if not os.path.exists(config_path):
        return [gr.update()]*7

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return [
            gr.update(value=config_data.get("language")),
            gr.update(value=config_data.get("voice_mode")),
            gr.update(value=config_data.get("clone_source")),
            gr.update(value=config_data.get("library_voice")),
            gr.update(value=config_data.get("stock_voice")),
            gr.update(value=config_data.get("output_format")),
            gr.update(value=config_data.get("srt_timing_mode"))
        ]
    except Exception as e:
        print(f"Error loading config {config_name}: {e}")
        return [gr.update()]*7

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

def load_whisper_model(model_size, device):
    global whisper_model, current_whisper_device, current_whisper_model_size
    if whisper_model is not None and current_whisper_model_size == model_size and current_whisper_device == device:
        return "Whisper model is already loaded."
    print(f"⏳ Loading Whisper model '{model_size}' to device: {device}...")
    try:
        whisper_model = whisper.load_model(model_size, device=device)
        current_whisper_device = device
        current_whisper_model_size = model_size
        print(f"✅ Whisper Model loaded successfully on {device}.")
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}")
    return "Whisper model is ready."

# ========================================================================================
# --- Core Logic Functions (unchanged) ---
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

def create_voiceover(segments, output_path, tts_instance, speaker_wav, language, sample_rate, timed_generation=True, strict_timing=False, progress=None):
    all_audio_chunks = []
    if timed_generation and strict_timing:
        total_duration_seconds = segments[-1].end.total_seconds() if segments else 0
        total_duration_samples = int(total_duration_seconds * sample_rate)
        final_audio = np.zeros(total_duration_samples, dtype=np.float32)
    elif timed_generation and not strict_timing:
        current_time_seconds = 0.0

    for i, sub in enumerate(segments):
        if progress: progress((i + 1) / len(segments), desc=f"Processing segment {i+1}/{len(segments)}")
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
            print(f"\n⚠️ Error generating TTS for line {sub.index}: '{raw_text}'. Skipping. Error: {e}")
            raise e
            
    if not timed_generation or (timed_generation and not strict_timing):
        final_audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)
    
    sf.write(output_path, final_audio, sample_rate)
    print("✅ Voiceover generation complete!")
    return output_path

# ========================================================================================
# --- Gradio Processing Functions ---
# ========================================================================================

def run_tts_generation(
    input_file, language, voice_mode, clone_source, library_voice, clone_speaker_audio, stock_voice,
    output_format, srt_timing_mode, tts_device, progress=gr.Progress(track_tqdm=True)
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
        file_extension = os.path.splitext(input_filepath)[1].lower()
        is_timed = file_extension in ['.srt', '.vtt']
        segments = parse_subtitle_file(input_filepath) if is_timed else parse_text_file(input_filepath)

        if not segments: return None, "🤷 No processable content found."

        output_dir = "gradio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_filepath))[0]
        clone_or_stock_name = library_voice if (voice_mode == 'Clone' and clone_source == 'Use from Library') else ("clone" if voice_mode == 'Clone' else stock_voice)
        output_filename = f"{base_name}_{clone_or_stock_name}_{language}.{output_format}"
        final_output_path = os.path.join(output_dir, output_filename)
        
        create_voiceover(
            segments=segments, output_path=final_output_path, tts_instance=tts_model,
            speaker_wav=speaker_wav_for_tts, language=language, sample_rate=SAMPLE_RATE,
            timed_generation=is_timed, strict_timing=(srt_timing_mode == "Strict (Cut audio to fit)"),
            progress=progress
        )
        
        return final_output_path, f"✅ Success! Audio saved to {final_output_path}"

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, f"❌ An unexpected error occurred: {e}"

def run_whisper_transcription(
    audio_file_path, model_size, language, task, output_action, whisper_device, progress=gr.Progress(track_tqdm=True)
):
    if audio_file_path is None: return "❌ Error: Please upload an audio file.", "", [], gr.update()
    try:
        progress(0, desc="Loading Whisper Model...")
        load_whisper_model(model_size, whisper_device)
        lang = language if language and language.strip() else None
        progress(0.2, desc=f"Starting {task}...")
        result = whisper_model.transcribe(audio_file_path, language=lang, task=task, verbose=True)
        full_text = result['text']
        
        if output_action == "Display Only": return full_text, "", [], gr.update()

        output_dir, base_name, timestamp = "whisper_outputs", os.path.splitext(os.path.basename(audio_file_path))[0], datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        txt_content = full_text
        srt_segments = [srt.Subtitle(i+1, timedelta(seconds=seg['start']), timedelta(seconds=seg['end']), seg['text'].strip()) for i, seg in enumerate(result['segments'])]
        srt_content = srt.compose(srt_segments)
        vtt_content = "WEBVTT\n\n" + "\n\n".join(f"{timedelta(seconds=seg['start'])} --> {timedelta(seconds=seg['end'])}\n{seg['text'].strip()}" for seg in result['segments'])
        json_content = json.dumps(result, indent=2, ensure_ascii=False)

        if output_action == "Save All Formats (.txt, .srt, .vtt, .json)":
            file_paths = []
            for ext, content in [("txt", txt_content), ("srt", srt_content), ("vtt", vtt_content), ("json", json_content)]:
                filepath = os.path.join(output_dir, f"{base_name}_{timestamp}.{ext}")
                with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
                file_paths.append(filepath)
            return full_text, srt_content, file_paths, gr.update()

        elif "Pipeline" in output_action:
            ext = "txt" if output_action == "Pipeline .txt to TTS" else "srt"
            content = txt_content if ext == "txt" else srt_content
            pipeline_filepath = os.path.join(output_dir, f"{base_name}_{timestamp}_pipelined.{ext}")
            with open(pipeline_filepath, 'w', encoding='utf-8') as f: f.write(content)
            return full_text, content, [pipeline_filepath], gr.update(value=pipeline_filepath)

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return f"❌ An unexpected error occurred: {e}", "", [], gr.update()

# ========================================================================================
# --- Gradio UI ---
# ========================================================================================

def create_gradio_ui():
    with gr.Blocks(title="Coqui XTTS & Whisper Pipeline") as demo:
        gr.HTML("""<div style="text-align: center; max-width: 800px; margin: 0 auto;"><h1 style="color: #4CAF50;">Coqui XTTS & Whisper Pipeline</h1><p style="font-size: 1.1em;">A complete toolkit for audio transcription and voiceover generation.</p></div>""")
        
        with gr.Accordion("⚙️ Global Device Settings", open=False):
            with gr.Row():
                tts_device = gr.Radio(label="TTS Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])
                whisper_device = gr.Radio(label="Whisper Device", choices=AVAILABLE_DEVICES, value=AVAILABLE_DEVICES[0])

        with gr.Tabs() as tabs:
            with gr.Tab("Whisper Transcription", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 1. Upload Audio")
                        whisper_audio_input = gr.Audio(label="Input Audio", type="filepath")
                        gr.Markdown("## 2. Configure Transcription")
                        whisper_model_size = gr.Dropdown(label="Whisper Model", choices=["tiny", "base", "small", "medium", "large", "turbo"], value="base")
                        whisper_language = gr.Textbox(label="Language (optional)", info="e.g., 'en', 'es'. Leave blank to auto-detect.")
                        whisper_task = gr.Radio(label="Task", choices=["transcribe", "translate"], value="transcribe")
                        gr.Markdown("## 3. Choose Output Action")
                        whisper_output_action = gr.Radio(label="Action", choices=["Display Only", "Save All Formats (.txt, .srt, .vtt, .json)", "Pipeline .txt to TTS", "Pipeline .srt to TTS"], value="Display Only")
                        
                        with gr.Group(visible=False) as whisper_pipeline_group:
                            whisper_autorun_tts = gr.Checkbox(label="Auto-run TTS after pipeline", value=False)
                            whisper_tts_config = gr.Dropdown(label="TTS Config to use", choices=get_config_files())
                            whisper_refresh_configs_btn = gr.Button("Refresh Configs")

                        transcribe_btn = gr.Button("Transcribe Audio", variant="primary")

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
                            tts_load_config_dd = gr.Dropdown(label="Load Config", choices=get_config_files())
                            tts_load_config_btn = gr.Button("Load Selected Config")
                            tts_refresh_configs_btn = gr.Button("Refresh Configs")
                            tts_config_save_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("## 1. Upload Your Content")
                        tts_input_file = gr.File(label="Input File (.txt, .srt, .vtt)", file_types=['.txt', '.srt', '.vtt'])
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
                        with gr.Accordion("Advanced SRT Settings", open=True): tts_srt_timing_mode = gr.Radio(label="SRT Timing Mode", choices=["Strict (Cut audio to fit)", "Flexible (Prevent audio cutoff)"], value="Flexible (Prevent audio cutoff)")
                        tts_generate_btn = gr.Button("Generate Voiceover", variant="primary")

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
        
        # Whisper Tab
        def handle_whisper_model_change(model_choice):
            if model_choice == "turbo":
                info_text = "⚠️ **Note:** The `turbo` model does not support translation."
                return gr.update(value="transcribe", interactive=False), gr.update(value=info_text, visible=True)
            return gr.update(interactive=True), gr.update(visible=False)
        whisper_model_size.change(fn=handle_whisper_model_change, inputs=whisper_model_size, outputs=[whisper_task, whisper_info_box])

        def handle_output_action_change(action):
            return gr.update(visible="Pipeline" in action)
        whisper_output_action.change(fn=handle_output_action_change, inputs=whisper_output_action, outputs=whisper_pipeline_group)

        # TTS Tab
        def update_tts_voice_mode(mode): return { tts_clone_voice_group: gr.update(visible=mode == 'Clone'), tts_stock_voice_group: gr.update(visible=mode == 'Stock') }
        tts_voice_mode.change(fn=update_tts_voice_mode, inputs=tts_voice_mode, outputs=[tts_clone_voice_group, tts_stock_voice_group])
        
        def update_clone_source(source): return { tts_upload_group: gr.update(visible=source == 'Upload New Sample'), tts_library_group: gr.update(visible=source == 'Use from Library') }
        tts_clone_source.change(fn=update_clone_source, inputs=tts_clone_source, outputs=[tts_upload_group, tts_library_group])

        # Voice Library Tab
        def refresh_all_voice_lists():
            voices = get_library_voices()
            return gr.update(choices=voices), gr.update(value="\n".join(voices))
        lib_save_btn.click(fn=save_voice_to_library, inputs=[lib_new_voice_audio, lib_new_voice_name], outputs=[lib_save_status]).then(fn=refresh_all_voice_lists, outputs=[tts_library_voice, lib_voice_list])
        lib_refresh_btn.click(fn=refresh_all_voice_lists, outputs=[tts_library_voice, lib_voice_list])
        refresh_library_btn_tts.click(fn=refresh_all_voice_lists, outputs=[tts_library_voice, lib_voice_list])

        # Config Management
        def refresh_config_lists():
            configs = get_config_files()
            return gr.update(choices=configs), gr.update(choices=configs)
        tts_save_config_btn.click(fn=save_tts_config, inputs=[tts_config_name, tts_language, tts_voice_mode, tts_clone_source, tts_library_voice, tts_stock_voice, tts_output_format, tts_srt_timing_mode], outputs=tts_config_save_status).then(fn=refresh_config_lists, outputs=[tts_load_config_dd, whisper_tts_config])
        tts_load_config_btn.click(fn=load_tts_config, inputs=tts_load_config_dd, outputs=[tts_language, tts_voice_mode, tts_clone_source, tts_library_voice, tts_stock_voice, tts_output_format, tts_srt_timing_mode])
        tts_refresh_configs_btn.click(fn=refresh_config_lists, outputs=[tts_load_config_dd, whisper_tts_config])
        whisper_refresh_configs_btn.click(fn=refresh_config_lists, outputs=[tts_load_config_dd, whisper_tts_config])

        # --- Main Pipeline Logic ---
        def handle_transcription_and_pipeline(
            audio_file_path, model_size, language, task, output_action, whisper_device,
            autorun, tts_config, progress=gr.Progress(track_tqdm=True)
        ):
            text_out, preview_out, files_out, tts_input_file_val = run_whisper_transcription(
                audio_file_path, model_size, language, task, output_action, whisper_device, progress
            )
            
            if autorun and tts_input_file_val:
                progress(0.9, desc="Auto-running TTS...")
                config_path = os.path.join(CONFIG_LIBRARY_PATH, f"{tts_config}.json")
                if not os.path.exists(config_path):
                    return text_out, preview_out, files_out, tts_input_file_val, None, f"❌ Auto-run failed: Config '{tts_config}' not found."
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                tts_audio_out, tts_status_out = run_tts_generation(
                    input_file=tts_input_file_val,
                    language=config_data["language"],
                    voice_mode=config_data["voice_mode"],
                    clone_source=config_data["clone_source"],
                    library_voice=config_data["library_voice"],
                    clone_speaker_audio=None,
                    stock_voice=config_data["stock_voice"],
                    output_format=config_data["output_format"],
                    srt_timing_mode=config_data["srt_timing_mode"],
                    tts_device=tts_device.value,
                    progress=progress
                )
                return text_out, preview_out, files_out, tts_input_file_val, tts_audio_out, tts_status_out

            return text_out, preview_out, files_out, tts_input_file_val, None, ""

        transcribe_btn.click(
            fn=handle_transcription_and_pipeline,
            inputs=[whisper_audio_input, whisper_model_size, whisper_language, whisper_task, whisper_output_action, whisper_device, whisper_autorun_tts, whisper_tts_config],
            outputs=[whisper_output_text, whisper_file_preview, whisper_output_files, tts_input_file, tts_output_audio, tts_status_textbox]
        ).then(fn=lambda file: gr.update(selected=1) if file else gr.update(), inputs=tts_input_file, outputs=tabs)

        tts_generate_btn.click(
            fn=run_tts_generation,
            inputs=[
                tts_input_file, tts_language, tts_voice_mode, tts_clone_source, tts_library_voice,
                tts_clone_speaker_audio, tts_stock_voice, tts_output_format, tts_srt_timing_mode, tts_device
            ],
            outputs=[tts_output_audio, tts_status_textbox]
        )

    return demo

if __name__ == "__main__":
    app = create_gradio_ui()
    
    print("\n✅ Gradio UI created. Launching Web UI...")
    print("➡️ Access the UI by opening the 'Running on local URL' link below in your browser.")
    
    app.launch(share=True)
