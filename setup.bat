@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Generating sample data...
cd data
python generate_sample_data.py
cd ..

echo.
echo Setup complete!
echo.
echo To run the dashboard, execute: run_dashboard.bat
pause
