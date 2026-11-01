@echo off
set OPENBLAS_NUM_THREADS=1
echo ======================================================================
echo  Starting Hotel Dynamic Pricing Web Dashboard on http://127.0.0.1:8000
echo ======================================================================
start http://127.0.0.1:8000
"C:\Users\aarav\AppData\Local\Programs\Python\Python312\python.exe" server.py
pause

