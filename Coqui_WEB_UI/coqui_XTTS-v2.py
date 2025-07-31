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

# ✨ Import the necessary config classes for PyTorch's safelist
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# ========================================================================================
# --- Global Model and Configuration ---
# ========================================================================================

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_NAME_FOR_FILE = "Coqui_XTTSv2"
SAMPLE_RATE = 24000  # XTTS uses a 24kHz sample rate

# Global variables to hold the models
tts_model = None
whisper_model = None
USE_CUDA = torch.cuda.is_available()

# Dictionary of available stock voices and their public URLs (UPDATED)
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
# --- Model Loading Functions ---
# ========================================================================================

def load_tts_model():
    """Loads the Coqui TTS model into a global variable."""
    global tts_model
    if tts_model is None:
        print("✅ Python script started. Initializing Coqui XTTS v2 model...")
        print("⏳ This may take a long time on the first run as it downloads the model (several GB).")
        if not USE_CUDA:
            print("🟡 WARNING: CUDA not available for TTS, using CPU. This will be very slow.")
        else:
            print("✅ CUDA is available for TTS, using GPU.")
        
        try:
            tts_model = TTS(MODEL_NAME, gpu=USE_CUDA)
            print("✅ TTS Model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load Coqui TTS model. Error: {e}")
            raise RuntimeError(f"Failed to load TTS model: {e}")

def load_whisper_model(model_size):
    """Loads the Whisper model into a global variable."""
    global whisper_model
    print(f"⏳ Loading Whisper model: {model_size}...")
    if not USE_CUDA:
        print("🟡 WARNING: CUDA not available for Whisper, using CPU. Transcription will be slow.")
    
    try:
        whisper_model = whisper.load_model(model_size)
        print("✅ Whisper Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load Whisper model. Error: {e}")
        raise RuntimeError(f"Failed to load Whisper model: {e}")

# ========================================================================================
# --- Core Logic Functions ---
# ========================================================================================

def normalize_text(text):
    """Normalizes numbers in text to words."""
    text = re.sub(r'(\d+)%', lambda m: num2words(int(m.group(1))) + ' percent', text)
    text = re.sub(r'(\d+)x(\d+)', lambda m: f"{num2words(int(m.group(1)))} by {num2words(int(m.group(2)))}", text)
    text = re.sub(r'(\d+)[–—-](\d+)', lambda m: f"{num2words(int(m.group(1)))} to {num2words(int(m.group(2)))}", text)
    text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0))), text)
    return text

def parse_subtitle_file(path):
    """Parses .srt or .vtt files."""
    print(f"📖 Parsing subtitle file: {os.path.basename(path)}")
    with open(path, 'r', encoding='utf-8') as f:
        return list(srt.parse(f.read()))

def parse_text_file(path):
    """Parses a .txt file, treating each line as a segment."""
    print(f"📖 Parsing text file: {os.path.basename(path)}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return [srt.Subtitle(index=i, start=timedelta(0), end=timedelta(0), content=line) for i, line in enumerate(lines, 1)]

def create_voiceover(segments, output_path, tts_instance, speaker_wav, language, sample_rate, timed_generation=True, strict_timing=False, progress=None):
    """Generates a voiceover from a list of text segments."""
    all_audio_chunks = []
    
    print(f"🎤 Generating voiceover for {len(segments)} segments...")

    if timed_generation and strict_timing:
        total_duration_seconds = segments[-1].end.total_seconds() if segments else 0
        total_duration_samples = int(total_duration_seconds * sample_rate)
        final_audio = np.zeros(total_duration_samples, dtype=np.float32)
    elif timed_generation and not strict_timing:
        current_time_seconds = 0.0

    num_segments = len(segments)
    for i, sub in enumerate(segments):
        if progress:
            progress((i + 1) / num_segments, desc=f"Processing segment {i+1}/{num_segments}")

        raw_text = sub.content.strip().replace('\n', ' ')
        if not raw_text:
            continue
        
        text_to_speak = normalize_text(raw_text)

        try:
            audio_chunk = np.array(tts_instance.tts(
                text=text_to_speak,
                speaker_wav=speaker_wav,
                language=language,
                split_sentences=False
            ), dtype=np.float32)

            if timed_generation:
                if strict_timing:
                    start_time_sec = sub.start.total_seconds()
                    subtitle_duration_sec = sub.end.total_seconds() - start_time_sec
                    generated_duration_sec = len(audio_chunk) / sample_rate
                    if generated_duration_sec > subtitle_duration_sec:
                        warnings.warn(f"\nLine {sub.index}: Speech ({generated_duration_sec:.2f}s) is LONGER than subtitle duration ({subtitle_duration_sec:.2f}s) and will be CUT OFF.")
                    start_sample = int(start_time_sec * sample_rate)
                    end_sample = start_sample + len(audio_chunk)
                    if end_sample > len(final_audio):
                        audio_chunk = audio_chunk[:len(final_audio) - start_sample]
                        end_sample = len(final_audio)
                    final_audio[start_sample:end_sample] = audio_chunk
                else: # not strict_timing
                    chunk_duration_seconds = len(audio_chunk) / sample_rate
                    target_start_time = sub.start.total_seconds()
                    silence_duration = target_start_time - current_time_seconds
                    if silence_duration > 0:
                        silence_samples = int(silence_duration * sample_rate)
                        all_audio_chunks.append(np.zeros(silence_samples, dtype=np.float32))
                    subtitle_duration = sub.end.total_seconds() - sub.start.total_seconds()
                    if chunk_duration_seconds > subtitle_duration:
                         warnings.warn(f"\nLine {sub.index}: Speech ({chunk_duration_seconds:.2f}s) is LONGER than subtitle duration ({subtitle_duration:.2f}s). It may overlap.")
                    all_audio_chunks.append(audio_chunk)
                    current_time_seconds = target_start_time + chunk_duration_seconds
            else:
                all_audio_chunks.append(audio_chunk)
                silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
                all_audio_chunks.append(silence)
        except Exception as e:
            print(f"\n⚠️ Error generating TTS for line {sub.index}: '{raw_text}'. Skipping. Error: {e}")
            raise e
            
    if not timed_generation or (timed_generation and not strict_timing):
        final_audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)

    print(f"\n💾 Saving final audio to {output_path}...")
    sf.write(output_path, final_audio, sample_rate)
    print("✅ Voiceover generation complete!")
    return output_path

# ========================================================================================
# --- Gradio Processing Functions ---
# ========================================================================================

def run_tts_generation(
    input_file, language, voice_mode, clone_speaker_audio, stock_voice,
    output_format, srt_timing_mode, progress=gr.Progress(track_tqdm=True)
):
    if input_file is None:
        return None, "❌ Error: Please upload an input file (.txt, .srt, or .vtt)."
    if voice_mode == 'Clone' and clone_speaker_audio is None:
        return None, "❌ Error: Please upload a speaker audio file for cloning."

    try:
        progress(0, desc="Starting TTS Generation...")
        
        speaker_wav_for_tts = None
        if voice_mode == 'Clone':
            print(f"🗣️ Using CLONE mode. Speaker WAV: {os.path.basename(clone_speaker_audio)}")
            speaker_wav_for_tts = clone_speaker_audio
        elif voice_mode == 'Stock':
            if stock_voice not in STOCK_VOICES:
                raise ValueError(f"Invalid stock voice '{stock_voice}'.")
            print(f"🗣️ Using STOCK mode. Speaker: {stock_voice}")
            voice_url = STOCK_VOICES[stock_voice]
            print(f"⬇️ Downloading stock voice from: {voice_url}")
            speaker_wav_for_tts = pooch.retrieve(voice_url, known_hash=None, progressbar=True)
        
        if not os.path.exists(speaker_wav_for_tts):
            raise FileNotFoundError(f"Speaker reference file not found at '{speaker_wav_for_tts}'")

        input_filepath = input_file.name
        file_extension = os.path.splitext(input_filepath)[1].lower()
        segments = None
        is_timed = False

        if file_extension in ['.srt', '.vtt']:
            segments = parse_subtitle_file(input_filepath)
            is_timed = True
        elif file_extension == '.txt':
            segments = parse_text_file(input_filepath)
            is_timed = False
        else:
            raise ValueError(f"Unsupported file type '{file_extension}'. Please use .srt, .vtt, or .txt.")

        if not segments:
            return None, "🤷 No processable content found in the input file."

        output_dir = "gradio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_filepath))[0]
        clone_or_stock_name = "clone" if voice_mode == 'Clone' else stock_voice
        output_filename = f"{base_name}_{clone_or_stock_name}_{language}.{output_format}"
        final_output_path = os.path.join(output_dir, output_filename)
        
        print(f"Output will be saved to: {final_output_path}")

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
    audio_file_path, model_size, language, task, output_action, progress=gr.Progress(track_tqdm=True)
):
    if audio_file_path is None:
        return "❌ Error: Please upload an audio file.", None, None

    try:
        progress(0, desc="Loading Whisper Model...")
        load_whisper_model(model_size)
        
        lang = language if language and language.strip() else None

        progress(0.2, desc=f"Starting {task}...")
        result = whisper_model.transcribe(audio_file_path, language=lang, task=task, verbose=True)
        
        full_text = result['text']
        
        if output_action == "Display Only":
            return full_text, None, gr.update()

        output_dir = "whisper_outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_action == "Save to .txt file":
            txt_filename = f"{base_name}_{timestamp}.txt"
            txt_filepath = os.path.join(output_dir, txt_filename)
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(full_text)
            return f"✅ Saved to {txt_filepath}", txt_filepath, gr.update()

        elif output_action == "Save to .srt file":
            srt_filename = f"{base_name}_{timestamp}.srt"
            srt_filepath = os.path.join(output_dir, srt_filename)
            segments = []
            for i, seg in enumerate(result['segments']):
                start_time = timedelta(seconds=seg['start'])
                end_time = timedelta(seconds=seg['end'])
                segments.append(srt.Subtitle(index=i+1, start=start_time, end=end_time, content=seg['text'].strip()))
            with open(srt_filepath, 'w', encoding='utf-8') as f:
                f.write(srt.compose(segments))
            return f"✅ Saved to {srt_filepath}", srt_filepath, gr.update()

        elif output_action == "Pipeline to TTS":
            pipeline_filename = f"{base_name}_{timestamp}_pipelined.txt"
            pipeline_filepath = os.path.join(output_dir, pipeline_filename)
            with open(pipeline_filepath, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            return "✅ Pipelined to TTS tab.", None, gr.update(value=pipeline_filepath)

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return f"❌ An unexpected error occurred: {e}", None, gr.update()

# ========================================================================================
# --- Gradio UI ---
# ========================================================================================

def create_gradio_ui():
    with gr.Blocks(title="Coqui XTTS & Whisper Pipeline") as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 800px; margin: 0 auto;">
                <h1 style="color: #4CAF50;">Coqui XTTS & Whisper Pipeline</h1>
                <p style="font-size: 1.1em;">
                    A complete toolkit for audio transcription and voiceover generation.
                </p>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            with gr.Tab("Whisper Transcription", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 1. Upload Audio")
                        whisper_audio_input = gr.Audio(label="Input Audio", type="filepath")

                        gr.Markdown("## 2. Configure Transcription")
                        whisper_model_size = gr.Dropdown(
                            label="Whisper Model", 
                            choices=["tiny", "base", "small", "medium", "large", "turbo"], 
                            value="base"
                        )
                        whisper_language = gr.Textbox(label="Language (optional)", info="e.g., 'en', 'es'. Leave blank to auto-detect.")
                        whisper_task = gr.Radio(label="Task", choices=["transcribe", "translate"], value="transcribe")
                        
                        gr.Markdown("## 3. Choose Output Action")
                        whisper_output_action = gr.Radio(
                            label="Action",
                            choices=["Display Only", "Save to .txt file", "Save to .srt file", "Pipeline to TTS"],
                            value="Display Only"
                        )
                        
                        transcribe_btn = gr.Button("Transcribe Audio", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("## Transcription Result")
                        whisper_output_text = gr.Textbox(label="Output Text", lines=15, interactive=False)
                        whisper_output_file = gr.File(label="Generated File", interactive=False)
                        whisper_info_box = gr.Markdown(visible=False)

            with gr.Tab("TTS Voiceover", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 1. Upload Your Content")
                        tts_input_file = gr.File(label="Input File (.txt, .srt, .vtt)", file_types=['.txt', '.srt', '.vtt'])

                        gr.Markdown("## 2. Configure Voice & Language")
                        tts_language = gr.Dropdown(label="Language", choices=SUPPORTED_LANGUAGES, value="en")
                        tts_voice_mode = gr.Radio(label="Voice Mode", choices=['Clone', 'Stock'], value='Stock')

                        with gr.Group(visible=False) as tts_clone_voice_group:
                            tts_clone_speaker_audio = gr.Audio(label="Upload Voice Sample for Cloning (6-30s)", type="filepath")
                        
                        with gr.Group(visible=True) as tts_stock_voice_group:
                            tts_stock_voice = gr.Dropdown(label="Stock Voice", choices=list(STOCK_VOICES.keys()), value='Clarabelle')
                        
                        gr.Markdown("## 3. Set Output Format")
                        tts_output_format = gr.Radio(label="Output Format", choices=['wav', 'mp3'], value='wav')

                        with gr.Accordion("Advanced SRT Settings", open=True):
                            tts_srt_timing_mode = gr.Radio(
                                label="SRT Timing Mode",
                                choices=["Strict (Cut audio to fit)", "Flexible (Prevent audio cutoff)"],
                                value="Flexible (Prevent audio cutoff)",
                                info="Choose how to handle audio that is longer than the subtitle duration."
                            )

                        tts_generate_btn = gr.Button("Generate Voiceover", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("## Generated Audio")
                        tts_output_audio = gr.Audio(label="Output", type="filepath", show_download_button=True)
                        tts_status_textbox = gr.Textbox(label="Status", interactive=False)
        
        # --- Event Handling ---
        
        # Whisper Tab Logic
        def handle_whisper_model_change(model_choice):
            if model_choice == "turbo":
                info_text = "⚠️ **Note:** The `turbo` model does not support translation. The task has been set to `transcribe`."
                return gr.update(value="transcribe", interactive=False), gr.update(value=info_text, visible=True)
            else:
                return gr.update(interactive=True), gr.update(visible=False)

        whisper_model_size.change(
            fn=handle_whisper_model_change,
            inputs=whisper_model_size,
            outputs=[whisper_task, whisper_info_box]
        )

        def pipeline_to_tts(filepath):
            if filepath:
                return gr.update(value=filepath), gr.update(selected=1)
            return gr.update(), gr.update()

        transcribe_btn.click(
            fn=run_whisper_transcription,
            inputs=[whisper_audio_input, whisper_model_size, whisper_language, whisper_task, whisper_output_action],
            outputs=[whisper_output_text, whisper_output_file, tts_input_file]
        ).then(
            fn=pipeline_to_tts,
            inputs=[tts_input_file],
            outputs=[tts_input_file, tabs]
        )

        # TTS Tab Logic
        def update_voice_mode(mode):
            is_clone = mode == 'Clone'
            return {
                tts_clone_voice_group: gr.update(visible=is_clone),
                tts_stock_voice_group: gr.update(visible=not is_clone)
            }

        tts_voice_mode.change(
            fn=update_voice_mode,
            inputs=tts_voice_mode,
            outputs=[tts_clone_voice_group, tts_stock_voice_group]
        )

        tts_generate_btn.click(
            fn=run_tts_generation,
            inputs=[
                tts_input_file, tts_language, tts_voice_mode, tts_clone_speaker_audio,
                tts_stock_voice, tts_output_format, tts_srt_timing_mode
            ],
            outputs=[tts_output_audio, tts_status_textbox]
        )
    return demo

if __name__ == "__main__":
    load_tts_model() # Load TTS model at startup
    app = create_gradio_ui()
    
    print("\n✅ TTS Model loaded. Launching Gradio Web UI...")
    print("➡️ Access the UI by opening the 'Running on local URL' link below in your browser.")
    
    app.launch(share=True)
