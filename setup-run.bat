@echo off
REM ============================================================================
REM  Combined script to set up the Conda environment, install dependencies,
REM  download models, and launch the Gradio WebUI.
REM
REM  - Installs Python dependencies from requirements.txt
REM  - Installs CUDA, cuDNN, and PyTorch
REM  - Installs Higgs Audio
REM  - Installs Coqui XTTS-v2 Model
REM ============================================================================

SET ENV_NAME=vox_env
SET PYTHON_VERSION=3.10
SET SCRIPT_NAME=Vox.py
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

REM --- Install PyTorch with CUDA support using the helper script ---
echo [INFO] Running the CUDA/cuDNN installation script...
CALL install_cuda.bat
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] The CUDA installation script failed. Aborting setup.
    pause
    exit /b 1
)
echo [INFO] CUDA/cuDNN installation complete.

REM --- Clone Higgs Audio repo if it doesn't exist ---
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

REM --- Ensure all other packages are installed FIRST ---
echo [INFO] Installing packages from %REQUIREMENTS_FILE%...
CALL conda run -n %ENV_NAME% pip install -r %REQUIREMENTS_FILE%
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install packages from %REQUIREMENTS_FILE%.
    pause
    exit /b 1
)
echo [INFO] All packages are installed/up-to-date.

REM --- Now, install Higgs Audio ---
echo [INFO] Installing Higgs Audio package...
CALL install_higgs.bat
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Higgs Audio properly.
    pause
    exit /b 1
)
echo [INFO] Higgs Audio installed successfully.

REM --- Download Coqui XTTS Model with Enhanced Error Checking ---
echo [INFO] Checking for Coqui XTTS model...
CALL install_coqui.bat
SET "COQUI_INSTALL_ERRORLEVEL=%ERRORLEVEL%"

echo [DEBUG] Returned from install_coqui.bat with ERRORLEVEL: %COQUI_INSTALL_ERRORLEVEL%

IF %COQUI_INSTALL_ERRORLEVEL% NEQ 0 (
    echo [ERROR] The Coqui XTTS model download script failed with error code %COQUI_INSTALL_ERRORLEVEL%.
    echo [ERROR] Please review any messages from the script above.
    pause
    exit /b 1
)
echo [INFO] Coqui XTTS model check complete.

REM --- Run the application using the reliable 'conda run' command ---
echo [INFO] Launching the Gradio WebUI in the '%ENV_NAME%' environment...
echo ============================================================================

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
echo [INFO] The application has finished. You can close this window.
pause
