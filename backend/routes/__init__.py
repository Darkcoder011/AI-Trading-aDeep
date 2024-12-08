from .upload import router as upload_router
from .analyze import router as analyze_router
from .analyze_models import router as analyze_models_router
from .websocket import router as websocket_router
from .prediction import router as prediction_router

__all__ = [
    'upload_router',
    'analyze_router',
    'analyze_models_router',
    'websocket_router',
    'prediction_router'
]
