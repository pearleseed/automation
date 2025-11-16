@echo off
REM Build script for Auto C-Peach using PyInstaller

echo ==========================================
echo Building Auto C-Peach Application
echo Target: Windows
echo ==========================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo WARNING: Virtual environment not detected.
    echo Activating venv...
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else (
        echo ERROR: Virtual environment not found!
        echo Please run: python -m venv venv
        pause
        exit /b 1
    )
)

echo Virtual environment: %VIRTUAL_ENV%
echo.

REM Check if PyInstaller is installed
where pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: PyInstaller not found. Installing...
    pip install pyinstaller
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo PyInstaller is installed
echo.

REM Check for required files
echo Checking required files...
if not exist "main.py" (
    echo ERROR: main.py not found!
    pause
    exit /b 1
)
echo   main.py found
echo.
echo Note: Data files (yolo11n.pt, data\, templates\) are NOT included in build.
echo       Place them next to the executable when distributing.

echo.
echo ==========================================
echo Building with PyInstaller...
echo ==========================================
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build using the spec file
pyinstaller --clean "Auto C-Peach.spec"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo Build completed successfully!
    echo ==========================================
    echo.
    echo Output location: .\dist\Auto C-Peach\
    echo.
    echo To run the application:
    echo   dist\Auto C-Peach\Auto C-Peach.exe
    echo.
) else (
    echo.
    echo ==========================================
    echo Build failed!
    echo ==========================================
    echo.
    echo Check the error messages above for details.
    pause
    exit /b 1
)

pause

