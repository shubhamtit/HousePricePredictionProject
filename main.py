from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("house_price_model.pkl")


class House(BaseModel):
    area_sqft : float
    bedrooms : int
    bathrooms : int
    age_years : int

@app.get('/')
def home():
    return {
        "message" : "House price prediction Api"
    }


@app.post('/prediction')
def predict_price(house : House):
    prediction = model.predict([
        [house.area_sqft,
        house.bedrooms,
        house.bathrooms,
        house.age_years]
    ])

    return {
        "predicted price" : prediction[0]
    }