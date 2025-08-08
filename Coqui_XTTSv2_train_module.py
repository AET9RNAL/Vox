import os
import gc
import sys
import tempfile
import traceback
import shutil
import torch
import torchaudio
import pandas
import subprocess
from faster_whisper import WhisperModel
from tqdm import tqdm
import gradio as gr
import re

# Check if 'trainer' is available, if not, define a dummy class to avoid import errors
# In a real environment, the 'trainer' package from TTS should be installed.
try:
    from trainer import Trainer, TrainerArgs
except ImportError:
    print("Warning: 'trainer' could not be imported. Using dummy classes. Ensure TTS is installed for full functionality.")
    class Trainer:
        def __init__(self, *args, **kwargs):
            self.output_path = "dummy_trainer_output"
        def fit(self): pass
        @staticmethod
        def init_from_config(config):
            print("Dummy Trainer initialized from config.")
            return Trainer()

    class TrainerArgs:
        def __init__(self, *args, **kwargs): pass


from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

# --- Global State ---
XTTS_MODEL = None

# --- Helper Functions ---
def clear_gpu_cache():
    """Clears the GPU cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def list_files(basePath, validExts=None, contains=None):
    """Lists files in a directory recursively."""
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()
            if validExts is None or ext.endswith(validExts):
                audioPath = os.path.join(rootDir, filename)
                yield audioPath

# --- Data Processing Backend (from formatter.py) ---
def format_audio_list(audio_files, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    """
    Processes a list of audio files into a formatted dataset for XTTS fine-tuning.
    It transcribes, segments, and creates training/evaluation metadata files.
    """
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Whisper Model for transcription...")
    compute_type = "float16" if device == "cuda" else "default"
    print(f"Using compute type: {compute_type}")
    asr_model = WhisperModel("large-v2", device=device, compute_type=compute_type)

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    tqdm_object = tqdm(audio_files)
    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting dataset...")

    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        segments = list(segments)
        words_list = []
        for segment in segments:
            words_list.extend(list(segment.words))

        first_word = True
        sentence = ""
        sentence_start_time = 0
        i = 0
        for word_idx, word in enumerate(words_list):
            if first_word:
                sentence_start_time = word.start
                if word_idx > 0:
                    previous_word_end = words_list[word_idx - 1].end
                    sentence_start_time = max(sentence_start_time - buffer, (previous_word_end + sentence_start_time) / 2)
                else:
                    sentence_start_time = max(sentence_start_time - buffer, 0)
                sentence = word.word.strip()
                first_word = False
            else:
                sentence += " " + word.word.strip()

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence.strip()
                if not sentence: continue
                
                sentence = multilingual_cleaners(sentence, target_language)
                
                audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
                output_filename = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"
                
                next_word_start = words_list[word_idx + 1].start if word_idx + 1 < len(words_list) else (wav.shape[0] / sr)
                word_end_time = min((word.end + next_word_start) / 2, word.end + buffer)
                
                absoulte_path = os.path.join(out_path, output_filename)
                os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
                i += 1
                first_word = True

                audio_segment = wav[int(sr * sentence_start_time):int(sr * word_end_time)].unsqueeze(0)
                if audio_segment.size(-1) >= sr / 3:
                    torchaudio.save(absoulte_path, audio_segment, sr)
                    metadata["audio_file"].append(output_filename)
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)
                
                sentence = ""

    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1).reset_index(drop=True)
    num_val_samples = int(len(df) * eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.sort_values('audio_file').to_csv(train_metadata_path, sep="|", index=False, header=True)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval.sort_values('audio_file').to_csv(eval_metadata_path, sep="|", index=False, header=True)

    del asr_model, df_train, df_eval, df, metadata
    gc.collect()
    clear_gpu_cache()

    return train_metadata_path, eval_metadata_path, audio_total_size

def preprocess_dataset(audio_files, language, out_path, progress=None):
    """Main entry point for dataset processing."""
    clear_gpu_cache()
    dataset_out_path = os.path.join(out_path, "dataset")
    os.makedirs(dataset_out_path, exist_ok=True)
    
    if not audio_files:
        return "You should provide one or multiple audio files!", "", ""
    
    try:
        audio_file_paths = [f.name for f in audio_files]
        train_meta, eval_meta, audio_total_size = format_audio_list(
            audio_file_paths, target_language=language, out_path=dataset_out_path, gradio_progress=progress
        )
    except Exception:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"Data processing failed. Check console for error: {error}", "", ""

    if audio_total_size < 120:
        message = "The total duration of the provided audio files should be at least 2 minutes."
        print(message)
        return message, "", ""

    print("Dataset processed successfully!")
    return "Dataset Processed!", train_meta, eval_meta

# --- Training Backend ---
def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995):
    """Sets up and runs the Coqui TTS GPTTrainer for fine-tuning."""
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    OUT_PATH = os.path.join(output_path, "run", "training")
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    config_dataset = BaseDatasetConfig(
        formatter="coqui", dataset_name="ft_dataset", path=os.path.dirname(train_csv),
        meta_file_train=os.path.basename(train_csv), meta_file_val=os.path.basename(eval_csv), language=language
    )
    
    dvae_checkpoint_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    mel_norm_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    tokenizer_file_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    xtts_checkpoint_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    xtts_config_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    dvae_checkpoint = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(dvae_checkpoint_link))
    mel_norm_file = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(mel_norm_link))
    tokenizer_file = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(tokenizer_file_link))
    xtts_checkpoint = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(xtts_checkpoint_link))
    xtts_config_file = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(xtts_config_link))

    files_to_download = [dvae_checkpoint_link, mel_norm_link, tokenizer_file_link, xtts_checkpoint_link, xtts_config_link]
    files_on_disk = [dvae_checkpoint, mel_norm_file, tokenizer_file, xtts_checkpoint, xtts_config_file]
    if not all(os.path.exists(f) for f in files_on_disk):
        print(" > Downloading original XTTS v2.0 model files...")
        ModelManager._download_model_files(files_to_download, CHECKPOINTS_OUT_PATH, progress_bar=True)

    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        debug_loading_failures=False, max_wav_length=max_audio_length,
        max_text_length=200, mel_norm_file=mel_norm_file,
        dvae_checkpoint=dvae_checkpoint, xtts_checkpoint=xtts_checkpoint,
        tokenizer_file=tokenizer_file, gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    
    config = GPTTrainerConfig(
        run_name=RUN_NAME, project_name=PROJECT_NAME, run_description="GPT XTTS fine-tuning",
        dashboard_logger="tensorboard", logger_uri=None, output_path=OUT_PATH,
        epochs=num_epochs, model_args=model_args, audio=audio_config,
        batch_size=batch_size, eval_batch_size=batch_size, batch_group_size=48,
        num_loader_workers=8, eval_split_max_size=256, print_step=50, plot_step=100,
        log_model_step=100, save_step=1000, save_n_checkpoints=1, save_checkpoints=True,
        print_eval=False, optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        [config_dataset], eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    trainer = Trainer(
        TrainerArgs(grad_accum_steps=grad_acumm, continue_path=None, restore_path=None),
        config, output_path=OUT_PATH, model=model,
        train_samples=train_samples, eval_samples=eval_samples,
    )
    trainer.fit()

    trainer_out_path = trainer.output_path
    
    dataset_path = os.path.join(os.path.dirname(train_csv), "wavs")
    samples_len = [os.path.getsize(os.path.join(dataset_path, item["audio_file"])) for item in train_samples]
    longest_audio_idx = samples_len.index(max(samples_len))
    speaker_ref = os.path.join(dataset_path, train_samples[longest_audio_idx]["audio_file"])

    del model, trainer, train_samples, eval_samples
    gc.collect()
    clear_gpu_cache()

    return xtts_config_file, tokenizer_file, trainer_out_path, speaker_ref

def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    """Main entry point for model training."""
    clear_gpu_cache()
    if not train_csv or not eval_csv:
        return "You need to run the data processing step first.", "", "", "", "", None
    
    try:
        max_audio_length_samples = int(max_audio_length * 22050)
        config_path, vocab_file, exp_path, speaker_wav = train_gpt(
            language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv,
            output_path=output_path, max_audio_length=max_audio_length_samples
        )
    except Exception:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"Training failed. Check console for error: {error}", "", "", "", "", None

    shutil.copy(config_path, exp_path)
    shutil.copy(vocab_file, exp_path)

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("Model training finished successfully!")
    clear_gpu_cache()
    
    log_dir = os.path.dirname(exp_path)
    return "Model training done!", os.path.join(exp_path, "config.json"), os.path.join(exp_path, "vocab.json"), ft_xtts_checkpoint, speaker_wav, log_dir

def launch_tensorboard(log_dir):
    """Launches TensorBoard in a new process."""
    if not log_dir or not os.path.exists(log_dir):
        return "Log directory not found. Please run training first.", gr.update(visible=False)
    
    command = f'tensorboard --logdir="{log_dir}"'
    print(f"Launching TensorBoard with command: {command}")

    subprocess.Popen(f'start cmd /k "{command}"', shell=True)
    
    url = "http://localhost:6006/"
    status_message = f"TensorBoard is starting in a new window. Command executed: {command}"
    url_markdown = f"If it doesn't open automatically, [click here to open TensorBoard]({url})."
    
    return status_message, gr.update(value=url_markdown, visible=True)


# --- Inference Backend ---


# --- Auto-discovery helpers for fine-tuned runs ---
import glob
from typing import Tuple

def _discover_latest_run(base_training_dir: str) -> Tuple[str, str, str, str]:
    """
    Return (config_path, vocab_path, checkpoint_path, speaker_ref) for the most recent run.
    Expects runs like: <base>/GPT_XTTS_FT-.../
    """
    pattern = os.path.join(base_training_dir, "GPT_XTTS_FT-*")
    candidates = glob.glob(pattern)
    if not candidates:
        return "", "", "", ""
    latest = max(candidates, key=os.path.getmtime)
    config_path = os.path.join(latest, "config.json")
    vocab_path = os.path.join(latest, "vocab.json")
    checkpoint_path = os.path.join(latest, "best_model.pth")
    speaker_ref = ""
    for cand in ("speaker_ref.wav", "speaker.wav", "reference.wav"):
        maybe = os.path.join(latest, cand)
        if os.path.exists(maybe):
            speaker_ref = maybe
            break
    return config_path, vocab_path, checkpoint_path, speaker_ref


def autofill_ft_paths(output_base_path: str):
    """
    Gradio callback: given Output Path (e.g., xtts_ft_training),
    find latest run under <out>/run/training and return paths.
    Returns: config, vocab, checkpoint, speaker_ref, message
    """
    if not output_base_path:
        return "", "", "", "", "Set 'Output Path' first."
    base_training_dir = os.path.join(output_base_path, "run", "training")
    c, v, ckpt, spk = _discover_latest_run(base_training_dir)
    if not all([c, v, ckpt]) or not all(os.path.exists(p) for p in (c, v, ckpt)):
        return "", "", "", "", f"No completed runs found under: {base_training_dir}"
    msg = f"Found latest run:\n{os.path.dirname(c)}"
    return c, v, ckpt, spk, msg



def load_or_discover_model(xtts_checkpoint: str, xtts_config: str, xtts_vocab: str, output_base_path: str):
    """
    Load using explicit paths if provided; otherwise try to discover latest run under output_base_path.
    Provides a valid dummy 'speakers_xtts.pth' to satisfy XTTS's internal path logic on Windows.
    """
    global XTTS_MODEL
    clear_gpu_cache()

    # Normalize Nones to empty strings for consistency
    xtts_checkpoint = xtts_checkpoint or ""
    xtts_config = xtts_config or ""
    xtts_vocab = xtts_vocab or ""

    # If any missing, try discovery
    if not all([xtts_checkpoint, xtts_config, xtts_vocab]):
        if not output_base_path:
            return "Error: Missing paths and Output Path is not set."
        base_training_dir = os.path.join(output_base_path, "run", "training")
        c, v, ckpt, _ = _discover_latest_run(base_training_dir)
        if not all([c, v, ckpt]) or not all(os.path.exists(p) for p in (c, v, ckpt)):
            return "Error: Missing one or more required paths (checkpoint, config, vocab). No run discovered."
        xtts_config, xtts_vocab, xtts_checkpoint = c, v, ckpt

    # Final validation
    if not (os.path.exists(xtts_config) and os.path.exists(xtts_vocab) and os.path.exists(xtts_checkpoint)):
        return "Error: One or more selected files do not exist on disk."

    # Prepare a safe speakers file path in the checkpoint directory
    checkpoint_dir = os.path.dirname(xtts_checkpoint) or os.getcwd()
    safe_speakers_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")
    try:
        if not os.path.exists(safe_speakers_path):
            # Create a tiny valid torch object so torch.load won't choke if accessed
            torch.save({}, safe_speakers_path)
    except Exception:
        # Fallback: put it next to the config file
        fallback_dir = os.path.dirname(xtts_config) or os.getcwd()
        safe_speakers_path = os.path.join(fallback_dir, "speakers_xtts.pth")
        try:
            torch.save({}, safe_speakers_path)
        except Exception:
            # Last resort: use os.devnull just to provide a truthy path (XTTS may not need to load it)
            safe_speakers_path = os.devnull

    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)

    print("Loading fine-tuned XTTS model (auto-discovery)...")
    XTTS_MODEL.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        vocab_path=xtts_vocab,
        speaker_file_path=safe_speakers_path,
        use_deepspeed=False,
    )

    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model loaded successfully!")
    return "Model Loaded!"

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    """Loads the fine-tuned XTTS model for inference."""
    global XTTS_MODEL
    clear_gpu_cache()
    
    if not all([xtts_checkpoint, xtts_config, xtts_vocab]):
        return "Error: Missing one or more required paths (checkpoint, config, vocab)."
    
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    
    print("Loading fine-tuned XTTS model...")
    # FIX: Explicitly set speaker_file_path to prevent an error when loading fine-tuned models.
    XTTS_MODEL.load_checkpoint(
        config, 
        checkpoint_path=xtts_checkpoint, 
        vocab_path=xtts_vocab, 
        speaker_file_path="",  # This prevents the library from looking for a non-existent file
        use_deepspeed=False
    )
    
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model loaded successfully!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file):
    """Runs TTS inference using the loaded fine-tuned model."""
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to load the model and provide a speaker audio file.", None, None

    try:
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio_file, 
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, 
            max_ref_length=XTTS_MODEL.config.max_ref_len, 
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs
        )
        
        out = XTTS_MODEL.inference(
            text=tts_text, language=lang, gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding, temperature=XTTS_MODEL.config.temperature,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k, top_p=XTTS_MODEL.config.top_p,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
            out_path = fp.name
            torchaudio.save(out_path, out["wav"], 24000)

        return "Speech generated successfully!", out_path, speaker_audio_file
    except Exception as e:
        traceback.print_exc()
        return f"Inference failed: {e}", None, None

# === Fine-tuning Inference Helpers ===
import srt

def _parse_srt_or_vtt(path):
    """Parses SRT or VTT file, supporting both ',' and '.' in timestamps."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Normalize decimal separator in timestamps to comma
    normalized = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1,\2', content)
    normalized = re.sub(r'(\d{2}:\d{2}:\d{2}),(\d{3})', r'\1,\2', normalized)  # ensure commas are preserved
    return list(srt.parse(normalized))

def _parse_text_as_srt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return [srt.Subtitle(index=i, start=timedelta(0), end=timedelta(0), content=line) for i, line in enumerate(lines, 1)]

def ft_inference_generate(
    ui_text_entry,
    txt_file_input,
    srt_vtt_file_input,
    input_mode,
    srt_timing_mode,
    language,
    speaker_wav,
    progress=gr.Progress()
):
    """
    Runs fine-tuned XTTS inference in 3 modes:
      1. UI Text Entry
      2. Regular Text File (.txt)
      3. SRT/VTT Subtitle File
    """
    try:
        from datetime import timedelta
        import numpy as np
        import soundfile as sf
        import srt

        global XTTS_MODEL
        if XTTS_MODEL is None:
            return None, "❌ Load the fine-tuned XTTS model first."

        # Determine input source & parsing logic
        segments = []
        timed_generation = False
        strict_timing = False

        if input_mode == "UI Text Entry":
            if not ui_text_entry or not ui_text_entry.strip():
                return None, "❌ Please enter some text."
            # One subtitle-like segment starting at t=0
            segments = [srt.Subtitle(index=1, start=timedelta(0), end=timedelta(0), content=ui_text_entry.strip())]

        elif input_mode == "Regular Text File (.txt)":
            if not txt_file_input:
                return None, "❌ Please upload a .txt file."
            file_path = txt_file_input.name if hasattr(txt_file_input, 'name') else txt_file_input
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            segments = [srt.Subtitle(index=i, start=timedelta(0), end=timedelta(0), content=line)
                        for i, line in enumerate(lines, 1)]

        elif input_mode == "SRT/VTT Subtitle File":
            if not srt_vtt_file_input:
                return None, "❌ Please upload an .srt or .vtt file."
            file_path = srt_vtt_file_input.name if hasattr(srt_vtt_file_input, 'name') else srt_vtt_file_input
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Normalize separators for parsing
            normalized = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1,\2', content)
            segments = list(srt.parse(normalized))
            timed_generation = True
            strict_timing = (srt_timing_mode == "Strict (Cut audio to fit)")

        if not segments:
            return None, "🤷 No content found."

        # Validate speaker_wav
        if not speaker_wav:
            return None, "❌ Please provide a speaker reference WAV file."
        speaker_wav_path = speaker_wav.name if hasattr(speaker_wav, 'name') else speaker_wav
        if not os.path.exists(speaker_wav_path):
            return None, f"❌ Speaker reference not found: {speaker_wav_path}"

        
        # Compute conditioning latents once for this session
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_wav_path,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
# Output setup
        output_dir = "gradio_outputs"
        os.makedirs(output_dir, exist_ok=True)
        mode_tag = re.sub(r"[^a-z0-9_]+", "_", input_mode.lower())
        output_path = os.path.join(output_dir, f"ft_inference_{mode_tag}.{ 'wav' }")

        SAMPLE_RATE = 24000
        all_audio_chunks = []

        if timed_generation and strict_timing:
            total_duration_seconds = segments[-1].end.total_seconds() if segments else 0
            total_duration_samples = int(total_duration_seconds * SAMPLE_RATE)
            final_audio = np.zeros(total_duration_samples, dtype=np.float32)
        elif timed_generation and not strict_timing:
            current_time_seconds = 0.0

        # Generate audio
        for i, sub in enumerate(segments):
            progress((i + 1) / len(segments), desc=f"TTS: Segment {i+1}/{len(segments)}")
            raw_text = sub.content.strip().replace('\n', ' ')
            if not raw_text:
                continue
            audio_chunk = np.asarray(XTTS_MODEL.inference(text=raw_text, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, temperature=XTTS_MODEL.config.temperature, length_penalty=XTTS_MODEL.config.length_penalty, repetition_penalty=XTTS_MODEL.config.repetition_penalty, top_k=XTTS_MODEL.config.top_k, top_p=XTTS_MODEL.config.top_p)["wav"], dtype=np.float32)
            if timed_generation:
                if strict_timing:
                    start_sample = int(sub.start.total_seconds() * SAMPLE_RATE)
                    end_sample = start_sample + len(audio_chunk)
                    if end_sample > len(final_audio):
                        audio_chunk = audio_chunk[:len(final_audio) - start_sample]
                        end_sample = len(final_audio)
                    final_audio[start_sample:end_sample] = audio_chunk
                else:
                    target_start_time = sub.start.total_seconds()
                    silence_duration = target_start_time - current_time_seconds
                    if silence_duration > 0:
                        all_audio_chunks.append(np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.float32))
                    all_audio_chunks.append(audio_chunk)
                    current_time_seconds = target_start_time + len(audio_chunk) / SAMPLE_RATE
            else:
                all_audio_chunks.append(audio_chunk)
                all_audio_chunks.append(np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32))

        if not timed_generation or (timed_generation and not strict_timing):
            final_audio = np.concatenate(all_audio_chunks) if all_audio_chunks else np.array([], dtype=np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, final_audio, SAMPLE_RATE)
        return output_path, f"✅ Success! Saved to {output_path}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error during inference: {e}"
