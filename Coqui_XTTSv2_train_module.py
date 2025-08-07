import os
import gc
import sys
import tempfile
import traceback
import torch
import torchaudio
import pandas
from faster_whisper import WhisperModel
from tqdm import tqdm

# Check if 'trainer' is available, if not, define a dummy class to avoid import errors
# In a real environment, the 'trainer' package from TTS should be installed.
try:
    from trainer import Trainer, TrainerArgs
except ImportError:
    print("Warning: 'trainer' could not be imported. Using dummy classes. Ensure TTS is installed for full functionality.")
    class Trainer:
        def __init__(self, *args, **kwargs): pass
        def fit(self): pass
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
    asr_model = WhisperModel("large-v2", device=device, compute_type="float16" if device == "cuda" else "default")

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
                sentence = word.word
                first_word = False
            else:
                sentence += word.word

            if word.word[-1] in ["!", ".", "?"]:
                sentence = sentence.strip()
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
                if audio_segment.size(-1) >= sr / 3:  # Ignore segments shorter than 0.33 seconds
                    torchaudio.save(absoulte_path, audio_segment, sr)
                    metadata["audio_file"].append(output_filename)
                    metadata["text"].append(sentence)
                    metadata["speaker_name"].append(speaker_name)

    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df) * eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.sort_values('audio_file').to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval.sort_values('audio_file').to_csv(eval_metadata_path, sep="|", index=False)

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
        train_meta, eval_meta, audio_total_size = format_audio_list(
            audio_files, target_language=language, out_path=dataset_out_path, gradio_progress=progress
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
        meta_file_train=train_csv, meta_file_val=eval_csv, language=language
    )
    
    # Download original model files if they don't exist
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
        epochs=num_epochs, output_path=OUT_PATH, model_args=model_args,
        run_name=RUN_NAME, project_name=PROJECT_NAME,
        run_description="GPT XTTS fine-tuning", dashboard_logger="tensorboard",
        audio=audio_config, batch_size=batch_size, batch_group_size=48,
        eval_batch_size=batch_size, num_loader_workers=8, eval_split_max_size=256,
        print_step=50, plot_step=100, log_model_step=100, save_step=1000,
        save_n_checkpoints=1, save_checkpoints=True, optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
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
        TrainerArgs(grad_accum_steps=grad_acumm),
        config, output_path=OUT_PATH, model=model,
        train_samples=train_samples, eval_samples=eval_samples,
    )
    trainer.fit()

    # Find a long audio file to use as a speaker reference for inference
    samples_len = [os.path.getsize(os.path.join(config_dataset.path, item["audio_file"])) for item in train_samples]
    longest_audio_idx = samples_len.index(max(samples_len))
    speaker_ref = os.path.join(config_dataset.path, train_samples[longest_audio_idx]["audio_file"])

    trainer_out_path = trainer.output_path
    del model, trainer, train_samples, eval_samples
    gc.collect()
    clear_gpu_cache()

    return xtts_config_file, xtts_checkpoint, tokenizer_file, trainer_out_path, speaker_ref

def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    """Main entry point for model training."""
    clear_gpu_cache()
    if not train_csv or not eval_csv:
        return "You need to run the data processing step first.", "", "", "", ""
    
    try:
        max_audio_length_samples = int(max_audio_length * 22050)
        config_path, _, vocab_file, exp_path, speaker_wav = train_gpt(
            language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv,
            output_path=output_path, max_audio_length=max_audio_length_samples
        )
    except Exception:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"Training failed. Check console for error: {error}", "", "", "", ""

    # Copy original config and vocab to the experiment path for portability
    shutil.copy(config_path, exp_path)
    shutil.copy(vocab_file, exp_path)

    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    print("Model training finished successfully!")
    clear_gpu_cache()
    return "Model training done!", os.path.join(exp_path, "config.json"), os.path.join(exp_path, "vocab.json"), ft_xtts_checkpoint, speaker_wav

# --- Inference Backend (from xtts_demo.py) ---
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
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    
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
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
            out_path = fp.name
            torchaudio.save(out_path, out["wav"], 24000)

        return "Speech generated successfully!", out_path, speaker_audio_file
    except Exception as e:
        traceback.print_exc()
        return f"Inference failed: {e}", None, None
