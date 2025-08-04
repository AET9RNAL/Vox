@echo off
REM ============================================================================
REM Coqui XTTS-v2 model loader script from Hugging Face.
REM This script requires Git and Git LFS to be installed and in the system PATH.
REM ============================================================================

SET "MODEL_DIR=XTTS_model"
SET "REPO_URL=https://huggingface.co/coqui/XTTS-v2"

echo [INFO] This script will download the Coqui XTTS-v2 model.
echo [INFO] This is a large download (approx. 2.2 GB). Please be patient.
echo.

REM --- Check for Git ---
git --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not found in your system's PATH.
    echo [SOLUTION] Please install Git from https://git-scm.com/downloads and ensure it's added to your PATH.
    pause
    exit /b 1
)
echo [INFO] Git is found.

REM --- Check for Git LFS ---
git-lfs --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git LFS ^(Large File Storage^) is not found.
    echo [SOLUTION] 1. Download and install Git LFS from https://git-lfs.github.com/
    echo [SOLUTION] 2. After installation, open a new command prompt and run 'git lfs install' to initialize it.
    pause
    exit /b 1
)
echo [INFO] Git LFS is found.

REM --- Check if the model directory already exists ---
IF EXIST "%MODEL_DIR%" (
    echo [INFO] Model directory '%MODEL_DIR%' already exists.
    echo [INFO] Skipping download. If you want to re-download, please delete the '%MODEL_DIR%' folder and run this script again.
    goto :eof
)

REM --- Clone the repository using Git LFS ---
echo [INFO] Cloning the model repository from Hugging Face...
echo [INFO] This may take a while, please do not close this window.
git clone %REPO_URL% %MODEL_DIR%
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to clone the repository.
    echo [TROUBLESHOOTING] 1. Check your internet connection.
    echo [TROUBLESHOOTING] 2. Ensure you have run 'git lfs install' at least once in a command prompt.
    echo [TROUBLESHOOTING] 3. Try running this script again.
    
    REM Clean up partially created directory on failure
    IF EXIST "%MODEL_DIR%" (
        rmdir /s /q "%MODEL_DIR%"
    )
    pause
    exit /b 1
)

echo.
echo [SUCCESS] The Coqui XTTS-v2 model has been downloaded to the '%MODEL_DIR%' folder.

:eof
