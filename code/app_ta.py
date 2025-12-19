from fastapi import Depends, File, Form, Request, FastAPI, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
from model_registry import ModelRegistry
from helper import single_prediction, model_evaluation

BASE_DIR = Path(__file__).resolve().parent
DESCRIPTOR_PATH = BASE_DIR / "models" / "descriptor_ta.json"
TARGET_COLUMN = "crash_type"

templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry(DESCRIPTOR_PATH)
    registry.load()
    app.state.registry = registry
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('index.html', 
    {'request': request, 'project_name': 'Traffic Accidents', 'target': TARGET_COLUMN })

@app.get("/api/models")
def read_models(registry: ModelRegistry = Depends(get_registry)):
    return {
        "models": registry.list_models(),
        "default": registry.default_model,
    }

@app.post("/api/predict-single")
def predict_single(file: UploadFile = File(...),
                   model: str | None = Form(...),
                   registry: ModelRegistry = Depends(get_registry)):
    # performs prediction for a single record
    result = single_prediction(file, model, registry)
    return result

@app.post("/api/evaluate-models")
def evaluate_models(file: UploadFile = File(...),
                    registry: ModelRegistry = Depends(get_registry)):
    # evaluates all registered models on a dataset
    results = model_evaluation(file, registry, target=TARGET_COLUMN)
    return {"results": results}