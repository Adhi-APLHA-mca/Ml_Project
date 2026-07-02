import traceback
import logging
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.pipeline.training import train_model
from src.pipeline.prediction import predict_from_input

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/train")
def train(request: Request):
    try:
        model_path = train_model()
        return {"status": "success", "model_path": model_path}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/predict")
def predict(
    request: Request,
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...),
):
    try:
        prediction = predict_from_input(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )
        return {"prediction": prediction}
    except FileNotFoundError as exc:
        logging.error("Prediction missing artifact", exc_info=True)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logging.error("Prediction exception", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
