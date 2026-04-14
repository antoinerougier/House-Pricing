import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.staticfiles import StaticFiles

from src.model import load_model

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="House Pricing API")


# Route pour servir index.html
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


app.mount("/static", StaticFiles(directory="static"), name="static")

model = load_model("model.joblib")
scaler = joblib.load("scaler.joblib")


class HouseFeatures(BaseModel):
    GrLivArea: float
    TotalBsmtSF: float
    OverallQual: float
    YearBuilt: float
    YearRemodAdd: float
    FirstFlrSF: float  # 1stFlrSF
    SecondFlrSF: float  # 2ndFlrSF
    FullBath: float
    BedroomAbvGr: float
    GarageCars: float
    TotRmsAbvGrd: float
    Fireplaces: float


@app.get("/")
def root():
    return {"message": "House Pricing API — POST /predict pour estimer un prix"}


@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array(
        [
            [
                features.GrLivArea,
                features.TotalBsmtSF,
                features.OverallQual,
                features.YearBuilt,
                features.YearRemodAdd,
                features.FirstFlrSF,
                features.SecondFlrSF,
                features.FullBath,
                features.BedroomAbvGr,
                features.GarageCars,
                features.TotRmsAbvGrd,
                features.Fireplaces,
            ]
        ]
    )

    data_scaled = scaler.transform(data)
    price = model.predict(data_scaled)[0]
    return {"estimated_price": round(float(price), 2)}
