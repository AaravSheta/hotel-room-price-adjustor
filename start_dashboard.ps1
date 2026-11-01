$env:OPENBLAS_NUM_THREADS = "1"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Starting Hotel Dynamic Pricing Web Dashboard on http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8000"
& "C:\Users\aarav\AppData\Local\Programs\Python\Python312\python.exe" (Join-Path $PSScriptRoot "server.py")

