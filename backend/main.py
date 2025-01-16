from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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