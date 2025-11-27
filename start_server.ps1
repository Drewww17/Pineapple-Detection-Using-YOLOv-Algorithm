# start_server.ps1
$ErrorActionPreference = "Stop"
cd "$PSScriptRoot\server"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  py -3 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

$need = -not (python - << 'PY'
import importlib, sys
sys.exit(0 if importlib.util.find_spec('uvicorn') else 1)
PY
$LASTEXITCODE)

if ($need) {
  python -m pip install --upgrade pip
  pip install -r requirements.txt
}

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
