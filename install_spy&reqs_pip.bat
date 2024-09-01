@echo off
REM Set the name of the virtual environment folder
set "VENV_NAME=venv"

REM Check if the virtual environment already exists
if not exist %VENV_NAME% (
    echo Virtual environment "%VENV_NAME%" does not exist. Please create one manually, or run create_venv.bat provided.
    pause
    exit /b 1
)

REM Activate the virtual environment
call %VENV_NAME%\Scripts\activate

REM Inform the user
echo Virtual environment "%VENV_NAME%" is now activated.

REM Define the path to the requirements file relative to the batch file location then run
set "REQSPYFILE=%BASEDIR%spyder-dev-requirements.txt"
pip install -r "%REQSPYFILE%"

REM Install FluxCluster-specific Requirements
set "REQFLXFILE=%BASEDIR%requirements.txt"
pip install -r "%REQFLXFILE%"

REM Keep the command prompt open after activation
cmd /K

