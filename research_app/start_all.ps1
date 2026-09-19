# Start Qdrant
Write-Host "🚀 Starting Qdrant..." -ForegroundColor Green
cd c:\Users\srava\OneDrive\Documents\Desktop\LangGraph\research_app
docker-compose up -d
Start-Sleep -Seconds 5

# Check Qdrant health
Write-Host "🔍 Checking Qdrant health..." -ForegroundColor Yellow
curl http://localhost:6333/readyz

# Setup Python environment
Write-Host "`n🐍 Setting up Python environment..." -ForegroundColor Green
cd c:\Users\srava\OneDrive\Documents\Desktop\LangGraph

# Check if venv exists
if (-not (Test-Path "fastapi-env\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..."
    python -m venv fastapi-env
}

# Activate venv
& ".\fastapi-env\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt -q

# Start FastAPI
Write-Host "`n🚀 Starting FastAPI server..." -ForegroundColor Green
cd research_app
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Write-Host "`n✅ All services running!" -ForegroundColor Green
Write-Host "📊 Research UI: http://localhost:8000/research/ui" -ForegroundColor Cyan
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "🎛️  Qdrant Dashboard: http://localhost:6333/dashboard" -ForegroundColor Cyan
