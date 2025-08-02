from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from backend.inference.object_classifier import classify_object
from backend.inference.state_classifier import classify_state
from backend.inference.nlp_recommender import get_recipe_recommendations
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint: render index.html
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), description: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    object_class = classify_object(image)
    
    state = classify_state(image, object_class)
    
    description = f"{object_class} {state} " + description
    recipes = get_recipe_recommendations(description)
    
    recipes = recipes.to_dict(orient="records")
    return JSONResponse(recipes)
