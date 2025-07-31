from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import classify_object, classify_state, get_recipe_recommendations
from PIL import Image
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    object_class = classify_object(image)
    
    state = classify_state(image, object_class)
    
    description = f"{object_class} that is {state}"
    recipes = get_recipe_recommendations(description)
    
    return {
        "object_class": object_class,
        "state": state,
        "recipe_recommendations": recipes
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
