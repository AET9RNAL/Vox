@echo off
REM ============================================================================
REM Helper script to install TensorFlow for full TensorBoard functionality.
REM ============================================================================

SET ENV_NAME=vox_env

echo [INFO] Installing TensorFlow for TensorBoard...
CALL conda run -n %ENV_NAME% pip install tensorflow

REM --- Error Checking ---
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install TensorFlow.
    exit /b 1
)

echo [SUCCESS] TensorFlow installed successfully.
exit /b 0
