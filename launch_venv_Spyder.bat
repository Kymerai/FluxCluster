@echo off
REM Get the directory of this batch file to dynamically know where to run
set "BASEDIR=%~dp0"

REM Navigate to the directory where the venv is located (relative to the batch file location)
cd /d "%BASEDIR%venv"

REM Check if the virtual environment folder exists
if not exist Scripts\activate (
    echo Virtual environment not found in "%BASEDIR%venv".
    echo Please create the virtual environment before running this script.
    pause
    exit /b 1
)

REM Activate the virtual environment
call Scripts\activate

REM Check if Spyder is installed
where spyder >nul 2>&1
if %errorlevel% neq 0 (
    echo Spyder is not installed in the virtual environment.
    echo Please install Spyder using pip install spyder.
    pause
    exit /b 1
)

REM Launch Spyder and check if it launches successfully
start "" spyder
if %errorlevel% neq 0 (
    echo Failed to launch Spyder. Please check your installation.
    pause
    exit /b 1
)

REM If everything is successful, close the command prompt window automatically
exit