$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = (Get-Location).Path
$env:FLASK_APP = "app"
Write-Host "PYTHONPATH=$env:PYTHONPATH"
Write-Host "Starting DNE experiment on http://127.0.0.1:5000/?workerId=TEST001"
python -m flask run --host 127.0.0.1 --port 5000
