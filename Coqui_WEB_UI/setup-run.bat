@echo off
REM ============================================================================
REM  Final script to set up the Conda environment and launch the
REM  Coqui TTS Gradio WebUI.
REM ============================================================================

SET ENV_NAME=coqui_tts_env
SET PYTHON_VERSION=3.10
SET SCRIPT_NAME=coqui_XTTS-v2.py
SET REQUIREMENTS_FILE=requirements.txt

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

REM --- Check for and create the environment if needed ---
echo [INFO] Checking for Conda environment: %ENV_NAME%...
CALL conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [INFO] Environment not found. Creating it now...
    CALL conda create --name %ENV_NAME% python=%PYTHON_VERSION% -y
    
    echo [INFO] Verifying environment creation...
    CALL conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create or verify the Conda environment.
        pause
        exit /b 1
    )
    echo [INFO] Environment '%ENV_NAME%' created and verified successfully.
) ELSE (
    echo [INFO] Environment '%ENV_NAME%' already exists.
)

REM --- Ensure all packages are installed ---
echo [INFO] Installing/verifying packages from %REQUIREMENTS_FILE%. This may take a moment...
CALL conda run -n %ENV_NAME% pip install -r %REQUIREMENTS_FILE% >nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install packages from %REQUIREMENTS_FILE%.
    pause
    exit /b 1
)
echo [INFO] All packages are installed/up-to-date.

REM --- Run the application ---
echo [INFO] Launching the Gradio WebUI...
echo ============================================================================
CALL conda run -n %ENV_NAME% python %SCRIPT_NAME%
echo ============================================================================

echo.
echo [INFO] The application has exited. You can close this window.
pause
