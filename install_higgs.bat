@echo off
REM ============================================================================
REM Script to install Higgs Audio in the conda environment
REM ============================================================================

SET ENV_NAME=vox_env
SET HIGGS_AUDIO_DIR=higgs-audio

echo [INFO] Installing Higgs Audio...

REM Check if we're in the right directory
IF NOT EXIST "%HIGGS_AUDIO_DIR%" (
    echo [ERROR] higgs-audio directory not found in current location.
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

REM Change to higgs-audio directory
cd %HIGGS_AUDIO_DIR%

REM Clean any previous installation attempts
echo [INFO] Cleaning previous installation attempts...
CALL conda run -n %ENV_NAME% pip uninstall -y boson-multimodal
CALL conda run -n %ENV_NAME% pip uninstall -y higgs-audio

REM Install in editable mode
echo [INFO] Installing Higgs Audio in editable mode...
CALL conda run -n %ENV_NAME% pip install -e .
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Higgs Audio in editable mode.
    cd ..
    pause
    exit /b 1
)

REM Go back to project root
cd ..

REM Verify installation
echo [INFO] Verifying Higgs Audio installation...
CALL conda run -n %ENV_NAME% python -c "from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine; print('SUCCESS: Higgs Audio installed correctly')"
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Higgs Audio verification failed.
    pause
    exit /b 1
)

echo [SUCCESS] Higgs Audio has been properly installed.
