$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -m mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db --host 127.0.0.1 --port 5000 --workers 1
} finally {
    Pop-Location
}
