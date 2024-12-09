from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path
import logging
import datetime
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

logger = logging.getLogger(__name__)

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
@app.get("/")
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint that ensures the application is running properly
    """
    try:
        return {
            "status": "healthy",
            "service": "llm-market-analysis",
            "version": os.environ.get('RAILWAY_GIT_COMMIT_SHA', 'development'),
            "timestamp": str(datetime.datetime.utcnow()),
            "uptime": "ok"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "unhealthy", "error": str(e)}
        )

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

@app.get("/root")
async def root():
    return {"message": "Trading Analysis API"}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get port from environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )