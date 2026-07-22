@echo off
setlocal enableextensions
chcp 65001 >nul
title PDF 英文小说中文注释工具

rem =====================================================================
rem  One-click launcher.
rem  Creates a local Python virtual environment (first run only),
rem  installs dependencies, then opens the graphical annotator.
rem =====================================================================

cd /d "%~dp0"

set "VENV=%~dp0.venv"
set "PYEXE=%VENV%\Scripts\python.exe"

rem --- Locate a Python interpreter -------------------------------------
where py >nul 2>nul
if %errorlevel%==0 (
    set "BOOT=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "BOOT=python"
    ) else (
        echo.
        echo [!] 未检测到 Python。请先安装 Python 3.7 或更高版本：
        echo     https://www.python.org/downloads/windows/
        echo     安装时请勾选 "Add python.exe to PATH"。
        echo.
        pause
        exit /b 1
    )
)

rem --- Create the virtual environment on first run --------------------
if not exist "%PYEXE%" (
    echo [*] 首次运行：正在创建运行环境，请稍候…
    %BOOT% -m venv "%VENV%"
    if errorlevel 1 (
        echo [!] 创建虚拟环境失败。
        pause
        exit /b 1
    )
    "%PYEXE%" -m pip install --upgrade pip
    echo [*] 正在安装依赖（PyMuPDF, wordfreq, nltk, Pillow）…
    "%PYEXE%" -m pip install -e "%~dp0"
    if errorlevel 1 (
        echo [!] 安装依赖失败，请检查网络后重试。
        pause
        exit /b 1
    )
)

rem --- Launch the GUI --------------------------------------------------
echo [*] 启动注释工具…
"%PYEXE%" -m annotator.gui
if errorlevel 1 (
    echo.
    echo [!] 程序退出并报告了错误，请查看上方信息。
    pause
)

endlocal
