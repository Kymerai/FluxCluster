@echo off
REM Set the name of the virtual environment folder
set "VENV_NAME=venv"

REM Check if the virtual environment already exists
if exist %VENV_NAME% (
    echo Virtual environment "%VENV_NAME%" already exists.
    echo To activate it, run: "%VENV_NAME%\Scripts\activate" within the directory of the parent folder above the venv folder, or use the activate_venv.bat script
    pause
    exit /b 0
)

REM Create the virtual environment in the current directory
python -m venv %VENV_NAME%

REM Check if the virtual environment was created successfully
if exist %VENV_NAME% (
    echo Virtual environment "%VENV_NAME%" created successfully.
) else (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate the virtual environment
call %VENV_NAME%\Scripts\activate

REM Inform the user
echo Virtual environment "%VENV_NAME%" is now activated.

REM Keep the command prompt open after activation (optional)
pause
