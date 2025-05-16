from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("model.pkl")

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(request: Request,
                 sepal_length: float = Form(...),
                 sepal_width: float = Form(...),
                 petal_length: float = Form(...),
                 petal_width: float = Form(...)):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": f"Predicted class: {prediction}"
    })
