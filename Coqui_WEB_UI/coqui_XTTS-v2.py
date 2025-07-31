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
from datetime import timedelta
import pooch

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

# Global variable to hold the TTS model
tts_model = None
USE_CUDA = torch.cuda.is_available()

# Dictionary of available stock voices and their public URLs
STOCK_VOICES = {
    'Clarabelle': "https://coqui.gateway.scarf.sh/v1/models/xtts-v2/samples/female.wav",
    'Jordan': "https://coqui.gateway.scarf.sh/v1/models/xtts-v2/samples/male.wav",
    'Hina': "https://coqui.gateway.scarf.sh/v1/models/xtts-v2/samples/hina.wav",
    'William': "https://coqui.gateway.scarf.sh/v1/models/xtts-v2/samples/william.wav",
    'Grace': "https://coqui.gateway.scarf.sh/v1/models/xtts-v2/samples/grace.wav"
}

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
]

# ========================================================================================
# --- Core Logic Functions (from original script) ---
# ========================================================================================

def load_model():
    """Loads the Coqui TTS model into a global variable."""
    global tts_model
    if tts_model is None:
        print("Initializing Coqui XTTS v2 model...")
        if not USE_CUDA:
            print("WARNING: CUDA not available, using CPU. This will be very slow.")
        else:
            print("CUDA is available, using GPU.")
        
        # Add required classes to PyTorch's safelist
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
        try:
            tts_model = TTS(MODEL_NAME, gpu=USE_CUDA)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load Coqui TTS model. Error: {e}")
            # Propagate the error to the Gradio UI
            raise RuntimeError(f"Failed to load TTS model: {e}")

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

def create_voiceover(segments, output_path, tts_instance, speaker_wav, language, sample_rate, timed_generation=True, progress=None):
    """Generates a voiceover from a list of text segments, with progress tracking."""
    all_audio_chunks = []
    
    print(f"🎤 Generating voiceover for {len(segments)} segments...")
    
    if timed_generation:
        total_duration_seconds = segments[-1].end.total_seconds() if segments else 0
        total_duration_samples = int(total_duration_seconds * sample_rate)
        final_audio = np.zeros(total_duration_samples, dtype=np.float32)

    num_segments = len(segments)
    for i, sub in enumerate(segments):
        # Update Gradio progress bar
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
                split_sentences=True
            ), dtype=np.float32)
            
            if timed_generation:
                start_time_sec = sub.start.total_seconds()
                subtitle_duration_sec = sub.end.total_seconds() - start_time_sec
                generated_duration_sec = len(audio_chunk) / sample_rate

                if generated_duration_sec > subtitle_duration_sec:
                    warnings.warn(
                        f"\nLine {sub.index}: Speech ({generated_duration_sec:.2f}s) is LONGER than subtitle duration ({subtitle_duration_sec:.2f}s). "
                        f"It may overlap.\nText: \"{raw_text}\""
                    )

                start_sample = int(start_time_sec * sample_rate)
                if start_sample + len(audio_chunk) > len(final_audio):
                    audio_chunk = audio_chunk[:len(final_audio) - start_sample]
                final_audio[start_sample : start_sample + len(audio_chunk)] = audio_chunk
            else:
                all_audio_chunks.append(audio_chunk)
                silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
                all_audio_chunks.append(silence)

        except Exception as e:
            print(f"\n⚠️ Error generating TTS for line {sub.index}: '{raw_text}'. Skipping. Error: {e}")
            continue
            
    if not timed_generation:
        final_audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)

    print(f"\n💾 Saving final audio to {output_path}...")
    sf.write(output_path, final_audio, sample_rate)
    print("✅ Voiceover generation complete!")
    return output_path

# ========================================================================================
# --- Gradio Processing Function ---
# ========================================================================================

def run_tts_generation(
    input_file,
    language,
    voice_mode,
    clone_speaker_audio,
    stock_voice,
    output_format,
    progress=gr.Progress(track_tqdm=True)
):
    """The main function to be called by the Gradio interface."""
    
    # 1. --- Input Validation ---
    if input_file is None:
        return None, "❌ Error: Please upload an input file (.txt, .srt, or .vtt)."
    if voice_mode == 'Clone' and clone_speaker_audio is None:
        return None, "❌ Error: Please upload a speaker audio file for cloning."

    try:
        progress(0, desc="Starting TTS Generation...")
        
        # 2. --- Determine Speaker WAV ---
        speaker_wav_for_tts = None
        if voice_mode == 'Clone':
            print(f"🗣️ Using CLONE mode. Speaker WAV: {os.path.basename(clone_speaker_audio)}")
            speaker_wav_for_tts = clone_speaker_audio
        elif voice_mode == 'Stock':
            if stock_voice not in STOCK_VOICES:
                raise ValueError(f"Invalid stock voice '{stock_voice}'.")
            print(f"🗣️ Using STOCK mode. Speaker: {stock_voice}")
            voice_url = STOCK_VOICES[stock_voice]
            # Download the stock voice sample
            speaker_wav_for_tts = pooch.retrieve(voice_url, known_hash=None, progressbar=True)
        
        if not os.path.exists(speaker_wav_for_tts):
            raise FileNotFoundError(f"Speaker reference file not found at '{speaker_wav_for_tts}'")

        # 3. --- Parse Input File ---
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

        # 4. --- Determine Output Path ---
        output_dir = "gradio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_filepath))[0]
        clone_or_stock_name = "clone" if voice_mode == 'Clone' else stock_voice
        output_filename = f"{base_name}_{clone_or_stock_name}_{language}.{output_format}"
        final_output_path = os.path.join(output_dir, output_filename)
        
        print(f"Output will be saved to: {final_output_path}")

        # 5. --- Run Main Process ---
        create_voiceover(
            segments=segments,
            output_path=final_output_path,
            tts_instance=tts_model,
            speaker_wav=speaker_wav_for_tts,
            language=language,
            sample_rate=SAMPLE_RATE,
            timed_generation=is_timed,
            progress=progress
        )
        
        return final_output_path, f"✅ Success! Audio saved to {final_output_path}"

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, f"❌ An unexpected error occurred: {e}"

# ========================================================================================
# --- Gradio UI ---
# ========================================================================================

def create_gradio_ui():
    with gr.Blocks(title="Coqui XTTS Voiceover Generator") as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 800px; margin: 0 auto;">
                <h1 style="color: #4CAF50;">Coqui XTTS Voiceover Generator</h1>
                <p style="font-size: 1.1em;">
                    Create high-quality voiceovers from text or subtitle files using Coqui's powerful XTTS-v2 model.
                    Upload your file, choose a voice, and generate your audio.
                </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # --- Input Components ---
                gr.Markdown("## 1. Upload Your Content")
                input_file = gr.File(
                    label="Input File (.txt, .srt, .vtt)",
                    file_types=['.txt', '.srt', '.vtt']
                )

                gr.Markdown("## 2. Configure Voice & Language")
                language = gr.Dropdown(
                    label="Language",
                    choices=SUPPORTED_LANGUAGES,
                    value="en"
                )
                
                voice_mode = gr.Radio(
                    label="Voice Mode",
                    choices=['Clone', 'Stock'],
                    value='Stock'
                )

                with gr.Group(visible=False) as clone_voice_group:
                    clone_speaker_audio = gr.Audio(
                        label="Upload Voice Sample for Cloning (6-30s)",
                        type="filepath"
                    )
                
                with gr.Group(visible=True) as stock_voice_group:
                    stock_voice = gr.Dropdown(
                        label="Stock Voice",
                        choices=list(STOCK_VOICES.keys()),
                        value='Clarabelle'
                    )
                
                gr.Markdown("## 3. Set Output Format")
                output_format = gr.Radio(
                    label="Output Format",
                    choices=['wav', 'mp3'],
                    value='wav'
                )

                generate_btn = gr.Button("Generate Voiceover", variant="primary")

            with gr.Column(scale=2):
                # --- Output Components ---
                gr.Markdown("## Generated Audio")
                output_audio = gr.Audio(
                    label="Output",
                    type="filepath",
                    show_download_button=True
                )
                status_textbox = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown(
                    """
                    ### How to Use:
                    1.  **Upload File**: Provide a `.txt`, `.srt`, or `.vtt` file.
                        -   For `.txt` files, each line becomes a separate audio segment.
                        -   For `.srt`/`.vtt` files, the audio is timed to match the subtitles.
                    2.  **Configure Voice**:
                        -   **Stock**: Choose from a list of high-quality pre-made voices.
                        -   **Clone**: Upload a clean 6-30 second audio sample of a voice you want to clone.
                    3.  **Generate**: Click the button and wait for the process to complete. The generated audio will appear here.
                    """
                )

        # --- UI Logic ---
        def update_voice_mode(mode):
            is_clone = mode == 'Clone'
            return {
                clone_voice_group: gr.update(visible=is_clone),
                stock_voice_group: gr.update(visible=not is_clone)
            }

        voice_mode.change(
            fn=update_voice_mode,
            inputs=voice_mode,
            outputs=[clone_voice_group, stock_voice_group]
        )

        generate_btn.click(
            fn=run_tts_generation,
            inputs=[
                input_file,
                language,
                voice_mode,
                clone_speaker_audio,
                stock_voice,
                output_format
            ],
            outputs=[output_audio, status_textbox]
        )
    return demo

if __name__ == "__main__":
    # Load the model once at startup
    load_model()
    
    # Create and launch the Gradio UI
    app = create_gradio_ui()
    app.launch(share=True)

