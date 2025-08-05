@echo off
REM ============================================================================
REM StyleTTS2 model and dependency installer script.
REM This script will:
REM 1. Clone the StyleTTS2 repository.
REM 2. Download the pre-trained LibriTTS model from Hugging Face.
REM 3. Install Python dependencies from requirements_styletts2.txt.
REM ============================================================================

SET "ENV_NAME=vox_env"
SET "REPO_DIR=StyleTTS2"
SET "REPO_URL=https://github.com/yl4579/StyleTTS2.git"
SET "MODEL_DIR=StyleTTS2_models\LibriTTS"
SET "MODEL_URL=https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00100.pth"
SET "CONFIG_URL=https://huggingface.co/yl4579/StyleTTS2-LibriTTS/raw/main/Configs/config.yml"
SET "REQUIREMENTS_FILE=requirements_styletts2.txt"

echo [INFO] Starting StyleTTS2 Setup...

REM --- Check for Git ---
git --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not found in your system's PATH.
    echo [SOLUTION] Please install Git from https://git-scm.com/downloads and ensure it's added to your PATH.
    pause
    exit /b 1
)
echo [INFO] Git is found.

REM --- Clone the repository if it doesn't exist ---
IF NOT EXIST "%REPO_DIR%" (
    echo [INFO] Cloning the StyleTTS2 repository...
    git clone %REPO_URL% %REPO_DIR%
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to clone the StyleTTS2 repository. Check your internet connection.
        pause
        exit /b 1
    )
) ELSE (
    echo [INFO] StyleTTS2 repository already exists. Skipping clone.
)

REM --- Install Python dependencies ---
IF EXIST "%REQUIREMENTS_FILE%" (
    echo [INFO] Installing StyleTTS2 Python dependencies from %REQUIREMENTS_FILE%...
    CALL conda run -n %ENV_NAME% pip install -r %REQUIREMENTS_FILE%
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies for StyleTTS2.
        pause
        exit /b 1
    )
) ELSE (
    echo [WARNING] %REQUIREMENTS_FILE% not found. Skipping dependency installation.
)


REM --- Download the pre-trained model ---
echo [INFO] Checking for StyleTTS2 pre-trained model...
IF EXIST "%MODEL_DIR%\epochs_2nd_00100.pth" (
    echo [INFO] Pre-trained model already exists. Skipping download.
) ELSE (
    echo [INFO] Pre-trained model not found. Downloading...
    echo [INFO] This is a large download (approx. 500 MB). Please be patient.
    
    REM Create the directory for the model
    mkdir "%MODEL_DIR%" 2>nul
    
    REM Use curl to download the model and config
    echo [INFO] Downloading model checkpoint...
    curl -L %MODEL_URL% -o "%MODEL_DIR%\epochs_2nd_00100.pth" --progress-bar
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to download the model checkpoint. Check your internet connection.
        rmdir /s /q "%MODEL_DIR%"
        pause
        exit /b 1
    )
    
    echo [INFO] Downloading config file...
    curl -L %CONFIG_URL% -o "%MODEL_DIR%\config.yml" --progress-bar
     IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to download the config file.
        rmdir /s /q "%MODEL_DIR%"
        pause
        exit /b 1
    )
)

echo.
echo [SUCCESS] StyleTTS2 setup is complete.