from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load the saved Random Forest model
model = joblib.load("best_random_forest_model.pkl")

# Initialize FastAPI app
app = FastAPI(title="Insurance Cost Prediction API")

# Add CORS middleware (allow all origins for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema using Pydantic
class InsuranceInput(BaseModel):
    age: int = Field(..., ge=0)
    sex: str = Field(..., pattern="^(male|female)$")
    bmi: float = Field(..., ge=0)
    children: int = Field(..., ge=0)
    smoker: str = Field(..., pattern="^(yes|no)$")
    region_northeast: int = Field(..., ge=0, le=1)
    region_northwest: int = Field(..., ge=0, le=1)
    region_southeast: int = Field(..., ge=0, le=1)
    region_southwest: int = Field(..., ge=0, le=1)

# Define a POST endpoint for prediction
@app.post("/predict")
def predict_insurance_cost(data: InsuranceInput):
    # Convert categorical variables to numeric
    sex = 1 if data.sex == "male" else 0
    smoker = 1 if data.smoker == "yes" else 0

    # Prepare the feature array in the correct order
    features = np.array([[
        data.age,
        sex,
        data.bmi,
        data.children,
        smoker,
        data.region_northeast,
        data.region_northwest,
        data.region_southeast,
        data.region_southwest
    ]])

    # Make prediction
    predicted_cost = model.predict(features)[0]

    # Return the prediction as JSON
    return {"predicted_insurance_cost": float(predicted_cost)}