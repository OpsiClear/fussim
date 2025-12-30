@echo off
setlocal
echo Building wheel...
set "SCRIPT_DIR=%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
if errorlevel 1 (
    echo Failed to setup Visual Studio environment
    exit /b 1
)
REM Fix PATH to use MSVC linker instead of Git's link.exe
set PATH=%PATH:C:\Program Files\Git\usr\bin;=%
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE="
if defined VIRTUAL_ENV (
    set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
)

if not defined PYTHON_EXE (
    if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if errorlevel 1 (
        echo No Python interpreter found. Activate a venv or install Python.
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" setup.py bdist_wheel
echo Build completed with exit code %ERRORLEVEL%
