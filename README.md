# Vox: The All-in-One ASR&TTS AI Suite

A comprehensive toolkit for audio transcription and voice synthesis featuring Coqui XTTS-v2 and Higgs Audio TTS models.
Features
🎙️ Whisper Transcription

Multiple transcription engines (OpenAI Whisper, Stable-TS)
Advanced post-processing with segmentation, merging, and refinement
Multiple output formats (.txt, .srt, .vtt, .json)
Direct pipeline integration with TTS generation
Configuration management for reproducible workflows

🗣️ Coqui XTTS Voice Generation

Voice cloning from 6-30 second samples
Support for 17 languages with cross-language voice cloning
Stock voice library with pre-trained voices
Subtitle-to-speech conversion with timing modes
Custom voice library management

🔥 Higgs Audio TTS

State-of-the-art neural TTS with natural prosody
Multi-speaker dialogue generation
Long-form content synthesis with smart chunking
Subtitle generation with precise timing control
Advanced voice library system

🛠️ Fine-Tuning & Training

Custom XTTS model fine-tuning
Dataset preprocessing and validation
Training progress monitoring
Inference testing with trained models

Requirements
Microsoft build tools: https://visualstudio.microsoft.com/downloads
Conda: https://www.anaconda.com/download
FFmpeg: Required for audio processing
Download from FFmpeg.org
Add to system PATH

Python 3.10


To use Coqui fine tuning on Windows:
1. install CUDA Toolkit: https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network
2. Download cuDNN: https://developer.nvidia.com/rdp/cudnn-archive
3. Install cuDNN according to: https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html
4. Download cuBLAS and cuDNN from: https://github.com/Purfview/whisper-standalone-win/releases/tag/libs