@echo off
REM ============================================================================
REM Helper script to install CUDA Toolkit, cuDNN, and PyTorch into the
REM specified Conda environment. This avoids system-wide installations.
REM ============================================================================

SET ENV_NAME=vox_env

echo [INFO] Installing PyTorch, CUDA Toolkit, and cuDNN into the '%ENV_NAME%' environment.
echo [INFO] This is a large download and may take some time...

REM Use conda to install the packages from the correct channels.
REM Conda will resolve the dependencies to ensure compatible versions are installed.
CALL conda install -n %ENV_NAME% pytorch torchvision torchaudio pytorch-cuda=12.1 cudatoolkit cudnn -c pytorch -c nvidia -y

REM --- Error Checking ---
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch with CUDA/cuDNN.
    echo [TROUBLESHOOTING] 1. Ensure you are running this from an Anaconda Prompt.
    echo [TROUBLESHOOTING] 2. Check your internet connection.
    echo [TROUBLESHOOTING] 3. Make sure your NVIDIA drivers are up to date.
    exit /b 1
)

echo [SUCCESS] PyTorch, CUDA Toolkit, and cuDNN installed successfully into '%ENV_NAME%'.
exit /b 0
