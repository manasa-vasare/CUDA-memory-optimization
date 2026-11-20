@echo off
echo ===============================================================
echo  Running Experiment 6: Memory Optimization in CUDA Benchmarks
echo ===============================================================

if exist exp6.exe (
    set EXEPATH=exp6.exe
) else if exist bin\exp6.exe (
    set EXEPATH=bin\exp6.exe
) else (
    echo [ERROR] exp6.exe not found. Please run build.bat first!
    exit /b 1
)

echo [RUN] Executing with default sizes: Transpose N=4096, Matmul N=1024
%EXEPATH% %*
