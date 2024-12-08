from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path
from routes import (
    upload_router,
    analyze_router,
    analyze_models_router,
    websocket_router,
    prediction_router,
    llm_analysis,
    automl_analysis,
    llm_new
)

app = FastAPI(title="LLM Market Analysis Platform")

# CORS Configuration
origins = [
    "http://localhost:3000",  # React development server
    "https://localhost:3000",
    "*"  # Allow all origins in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint - this should be the first route
@app.get("/api/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "timestamp": os.environ.get('RAILWAY_GIT_COMMIT_SHA', 'development')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files from frontend build
static_files_dir = Path(__file__).parent.parent / "frontend" / "build"
if static_files_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_files_dir), html=True), name="static")

# Include routers
app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(analyze_models_router)
app.include_router(websocket_router)
app.include_router(prediction_router)
app.include_router(llm_analysis.router, prefix="/api/v1", tags=["llm"])
app.include_router(llm_new.router, prefix="/api/v1", tags=["llm"])
app.include_router(automl_analysis.router, prefix="/api/automl", tags=["automl"])

@app.get("/")
async def root():
    return {"message": "Trading Analysis API"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=65,
        workers=1
    )