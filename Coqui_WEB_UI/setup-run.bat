@echo off
REM ============================================================================
REM  Combined script to set up the Conda environment, install Higgs Audio,
REM  and launch the merged Coqui TTS & Higgs TTS Gradio WebUI.
REM
REM  FIX: Changed the final execution step to use 'conda activate' for better
REM  environment variable handling, matching the logic in run_with_conda.bat.
REM ============================================================================

SET ENV_NAME=coqui_tts_env
SET PYTHON_VERSION=3.10
SET SCRIPT_NAME=coqui_XTTS-v2.py
SET REQUIREMENTS_FILE=requirements.txt
SET HIGGS_AUDIO_DIR=higgs-audio
SET HIGGS_REPO_URL=https://github.com/boson-ai/higgs-audio.git

REM --- Check for required files ---
IF NOT EXIST "%SCRIPT_NAME%" (
    echo [ERROR] Cannot find the main application file: %SCRIPT_NAME%
    pause
    exit /b 1
)
IF NOT EXIST "%REQUIREMENTS_FILE%" (
    echo [ERROR] Cannot find the requirements file: %REQUIREMENTS_FILE%
    pause
    exit /b 1
)

REM --- Check if Conda is installed AND accessible ---
CALL conda --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda is not found in your system's PATH.
    echo [SOLUTION] Close this window and run this script from the 'Anaconda Prompt'.
    pause
    exit /b 1
)

REM --- Check if Git is installed ---
CALL git --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not found in your system's PATH.
    echo [SOLUTION] Please install Git from https://git-scm.com/downloads and ensure it's added to your PATH.
    pause
    exit /b 1
)

REM --- Check for and create the environment if needed ---
echo [INFO] Checking for Conda environment: %ENV_NAME%...
CALL conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Environment not found. Creating it now...
    CALL conda create --name %ENV_NAME% python=%PYTHON_VERSION% -y
    
    echo [INFO] Verifying environment creation...
    CALL conda run -n %ENV_NAME% python --version >nul 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create or verify the Conda environment.
        pause
        exit /b 1
    )
    echo [INFO] Environment '%ENV_NAME%' created and verified successfully.
) ELSE (
    echo [INFO] Environment '%ENV_NAME%' already exists.
)

REM --- Install PyTorch with CUDA support ---
echo [INFO] Installing PyTorch with CUDA support. This is a large download and may take some time...
CALL conda run -n %ENV_NAME% conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch with CUDA. Please check your internet connection and NVIDIA drivers.
    pause
    exit /b 1
)
echo [INFO] PyTorch installed successfully.

REM --- Clone and install Higgs Audio ---
IF NOT EXIST "%HIGGS_AUDIO_DIR%" (
    echo [INFO] Higgs Audio repository not found. Cloning from GitHub...
    CALL git clone %HIGGS_REPO_URL%
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to clone Higgs Audio repository.
        pause
        exit /b 1
    )
) ELSE (
    echo [INFO] Higgs Audio repository found.
)

echo [INFO] Installing Higgs Audio package properly...
CALL install_higgs.bat
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Higgs Audio properly.
    pause
    exit /b 1
)
echo [INFO] Higgs Audio installed successfully.

REM --- Ensure all other packages are installed ---
echo [INFO] Installing remaining packages from %REQUIREMENTS_FILE%...
CALL conda run -n %ENV_NAME% pip install -r %REQUIREMENTS_FILE%
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install packages from %REQUIREMENTS_FILE%.
    pause
    exit /b 1
)
echo [INFO] All packages are installed/up-to-date.

REM --- Run the application ---
echo [INFO] Activating Conda environment and launching the Gradio WebUI...
echo ============================================================================

REM *** FIX: Use 'conda activate' for more reliable environment setup ***
call conda activate %ENV_NAME%
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment: %ENV_NAME%
    echo [INFO] Make sure the environment exists by running: conda env list
    pause
    exit /b 1
)

python %SCRIPT_NAME%

echo ============================================================================

echo.
echo [INFO] The application has exited. You can close this window.
pause
