@echo off
REM -----------------------------------------------------
REM Setup Script for PR-Disaster-Tweets Project (Windows)
REM -----------------------------------------------------

REM Activate virtual environment (asumiendo que se llama .venv)
echo Activating virtual environment...
call .venv\Scripts\activate

REM Instalar dependencias desde requirements.txt
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

REM Descargar recursos NLTK necesarios
echo Downloading NLTK resources...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

echo.
echo Setup complete!
pause
