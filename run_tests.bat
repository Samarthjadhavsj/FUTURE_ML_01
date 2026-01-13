@echo off
echo ========================================
echo Running All Test Cases
echo ========================================
echo.

cd tests

echo Running Data Preprocessing Tests...
python test_preprocessing.py
echo.

echo Running Model Tests...
python test_model.py
echo.

echo Running Integration Tests...
python test_integration.py
echo.

echo ========================================
echo All Tests Complete!
echo ========================================
pause
