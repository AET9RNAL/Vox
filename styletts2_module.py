import gradio as gr
import torch
import os
import re
import yaml
import numpy as np
import soundfile as sf
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import subprocess
import sys
from datetime import datetime

# StyleTTS2 specific imports
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Modules.models import SynthesizerTrn

# --- Global Variables & Paths ---
STYLETTS_MODELS_PATH = "StyleTTS2_models"
FINETUNE_OUTPUT_PATH = "StyleTTS2_finetunes"
PRETRAINED_LIBRI_MODEL_URL = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00100.pth"
PRETRAINED_LIBRI_CONFIG_URL = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/raw/main/Configs/config.yml"

os.makedirs(STYLETTS_MODELS_PATH, exist_ok=True)
os.makedirs(FINETUNE_OUTPUT_PATH, exist_ok=True)

# Set espeak path for phonemizer
if sys.platform == "win32":
    # Attempt to find espeak-ng in common installation paths
    espeak_path = None
    for path in ["C:\\Program Files\\eSpeak NG\\espeak-ng.exe", "C:\\Program Files (x86)\\eSpeak NG\\espeak-ng.exe"]:
        if os.path.exists(path):
            espeak_path = path
            break
    if espeak_path:
        EspeakWrapper.set_library(espeak_path)
    else:
        print("WARNING: eSpeak NG not found in default paths. Phonemizer might not work.")
        print("Please ensure eSpeak NG is installed and its path is correctly configured if you face issues.")


# --- Model Loading ---
styletts_model = None
styletts_sampler = None
styletts_config = None
current_model_path = None

def load_styletts_model(model_path, config_path, device):
    global styletts_model, styletts_sampler, styletts_config, current_model_path
    
    if current_model_path == model_path and styletts_model is not None:
        return f"Model '{os.path.basename(model_path)}' is already loaded."

    print(f"Loading StyleTTS2 model from: {model_path}")
    if not os.path.exists(model_path):
        raise gr.Error(f"Model checkpoint not found at {model_path}")
    if not os.path.exists(config_path):
        raise gr.Error(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        styletts_config = yaml.load(f, Loader=yaml.FullLoader)

    model = SynthesizerTrn(
        styletts_config['data']['filter_length'] // 2 + 1,
        styletts_config['data']['segment_size'] // styletts_config['data']['hop_length'],
        **styletts_config['model']
    ).to(device)
    
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    
    styletts_model = model
    
    sampler = DiffusionSampler(
        model.diffusion.denoiser,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    styletts_sampler = sampler
    current_model_path = model_path
    
    print(f"Successfully loaded model '{os.path.basename(model_path)}' to {device}.")
    return f"Model '{os.path.basename(model_path)}' loaded successfully."

def get_available_models():
    models = {"Pre-trained LibriTTS": "Pre-trained LibriTTS"}
    if os.path.exists(FINETUNE_OUTPUT_PATH):
        for f in os.listdir(FINETUNE_OUTPUT_PATH):
            if f.endswith(".pth"):
                model_name = os.path.splitext(f)[0]
                models[f"Fine-tuned: {model_name}"] = os.path.join(FINETUNE_OUTPUT_PATH, f)
    return list(models.keys())

def get_model_paths(model_selection):
    if model_selection == "Pre-trained LibriTTS":
        model_path = os.path.join(STYLETTS_MODELS_PATH, "LibriTTS", "epochs_2nd_00100.pth")
        config_path = os.path.join(STYLETTS_MODELS_PATH, "LibriTTS", "config.yml")
        return model_path, config_path
    elif model_selection.startswith("Fine-tuned: "):
        model_name = model_selection.replace("Fine-tuned: ", "")
        model_path = os.path.join(FINETUNE_OUTPUT_PATH, f"{model_name}.pth")
        config_path = os.path.join(FINETUNE_OUTPUT_PATH, f"{model_name}.yml")
        return model_path, config_path
    return None, None

# --- Voice Generation ---
def run_styletts_generation(text, ref_audio, model_selection, alpha, beta, diffusion_steps, embedding_scale, device, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Please provide text to synthesize.")
    if ref_audio is None:
        raise gr.Error("Please provide a reference audio file for voice cloning.")

    model_path, config_path = get_model_paths(model_selection)
    if not model_path:
        raise gr.Error("Invalid model selection.")

    try:
        progress(0, desc="Loading StyleTTS2 Model...")
        load_styletts_model(model_path, config_path, device)

        progress(0.2, desc="Processing Reference Audio...")
        ref_path = ref_audio
        
        # Convert to WAV if necessary
        if not ref_path.lower().endswith(".wav"):
            print("Reference is not WAV, converting...")
            waveform, sr = torchaudio.load(ref_path)
            ref_path = "temp_ref.wav"
            torchaudio.save(ref_path, waveform, sr)

        wav, sr = torchaudio.load(ref_path)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        
        ref_spectrogram = styletts_model.get_style_latents(wav.to(device))

        progress(0.4, desc="Synthesizing Speech...")
        
        # Using the text_to_speech method from the model's utils
        output_wav = styletts_model.inference(
            text=text,
            style_latents=ref_spectrogram,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            sampler=styletts_sampler
        )
        
        progress(0.9, desc="Saving Audio...")
        output_dir = "styletts2_outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"styletts2_{timestamp}.wav"
        final_output_path = os.path.join(output_dir, output_filename)

        sf.write(final_output_path, output_wav, styletts_config['data']['sampling_rate'])
        
        return final_output_path, f"✅ Success! Audio saved to {final_output_path}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"An error occurred during generation: {e}")

# --- Fine-Tuning ---
def run_styletts_finetuning(dataset_path, model_name, epochs, batch_size, learning_rate, progress=gr.Progress()):
    if not dataset_path:
        raise gr.Error("Please provide a path to your prepared dataset.")
    if not model_name or not model_name.strip():
        raise gr.Error("Please provide a name for your new fine-tuned model.")

    sanitized_name = re.sub(r'[\\/*?:"<>|]', "", model_name).strip().replace(" ", "_")
    
    # This is a placeholder for the actual training script execution.
    # In a real scenario, this would call `train_finetune_accelerate.py`
    # with the correct arguments and a dynamically generated config.
    
    status = "Starting fine-tuning process...\n"
    status += f"Dataset: {dataset_path}\n"
    status += f"Model Name: {sanitized_name}\n"
    status += f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}\n\n"
    status += "NOTE: This is a simulation. In a real setup, this would launch a background training process.\n"
    
    # Simulate training progress
    for i in progress.tqdm(range(epochs), desc="Simulating Fine-Tuning"):
        import time
        time.sleep(2) # Simulate work for one epoch
    
    # Simulate creating model files
    dummy_model_path = os.path.join(FINETUNE_OUTPUT_PATH, f"{sanitized_name}.pth")
    dummy_config_path = os.path.join(FINETUNE_OUTPUT_PATH, f"{sanitized_name}.yml")
    with open(dummy_model_path, 'w') as f:
        f.write("This is a dummy model file.")
    with open(dummy_config_path, 'w') as f:
        f.write("# This is a dummy config file.")

    status += f"\n✅ Simulation complete. Model files would be saved as:\n- {dummy_model_path}\n- {dummy_config_path}"
    
    return status