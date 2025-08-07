# Vox: The All-in-One ASR&TTS AI Suite

A comprehensive toolkit for audio transcription and voice synthesis featuring Coqui XTTS-v2 and Higgs Audio TTS models.
Features

🎙️ Whisper Transcription

1. Multiple transcription engines (OpenAI Whisper, Stable-TS)
2. Advanced post-processing with segmentation, merging, and refinement
3. Multiple output formats (.txt, .srt, .vtt, .json)
4. Direct pipeline integration with TTS generation
5. Configuration management for reproducible workflows

🗣️ Coqui XTTS Voice Generation

1. Zero shot Voice cloning from 6-30 second samples
2. Model training & fine-tuning
3. Support for 17 languages with cross-language voice cloning
4. Stock voice library with pre-trained voices
5. Subtitle-to-speech conversion with timing modes
6. Custom voice library management

🔥 Higgs Audio TTS

1. State-of-the-art neural TTS with natural prosody
2. Multi-speaker dialogue generation
3. Long-form content synthesis with smart chunking
4. Subtitle generation with precise timing control
5. Advanced voice library system


Requirements
Microsoft build tools: https://visualstudio.microsoft.com/downloads

Conda: https://www.anaconda.com/download

FFmpeg: Required for audio processing

Download from FFmpeg.org

Add to system PATH

Python 3.10


To use Coqui fine tuning on Windows:
1. CUDA Toolkit and cuDNN will be installed automatically into the env