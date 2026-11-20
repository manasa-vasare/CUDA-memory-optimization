@echo off
echo ===============================================================
echo  Building Experiment 6: Memory Optimization in CUDA
echo ===============================================================

where nvcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] nvcc was not found in PATH!
    echo Please run this script inside the Developer Command Prompt for VS.
    exit /b 1
)

where cl >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] cl.exe was not found in PATH!
    echo Please open the Developer Command Prompt for VS 2022/2026.
    exit /b 1
)

if not exist bin mkdir bin

echo Compiling src\exp6.cu with -O2 optimization...
nvcc -O2 src\exp6.cu -o bin\exp6.exe

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Build succeeded: bin\exp6.exe
    copy /Y bin\exp6.exe exp6.exe >nul
    echo Copied executable to root: exp6.exe
) else (
    echo [FAILED] Compilation failed.
    exit /b %ERRORLEVEL%
)
